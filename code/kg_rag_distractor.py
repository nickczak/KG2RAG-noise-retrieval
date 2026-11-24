import os
import copy
import ujson as json
import argparse
import hashlib
import statistics
import tempfile
import time
from pathlib import Path
from tqdm import tqdm
from FlagEmbedding import FlagReranker
from llama_index.core import Settings,VectorStoreIndex,PromptTemplate
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from util.kg_post_processor import NaivePostprocessor,KGRetrievePostProcessor,ngram_overlap, GraphFilterPostProcessor, SemanticFilterPostProcessor, CrossEncoderFilterPostProcessor, JointFilterPostProcessor
from util.kg_response_synthesizer import get_response_synthesizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from util.score_reuse import (CachedReranker, NativeScoreStore, NativeScoreRetriever,
                             NativeSemanticScorePostprocessor, pinned_reranker)

import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('KGRAG')
logging.getLogger('KGRAG').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

from builtins import print as _print
from sys import _getframe
def print(*arg, **kw):
    s = f'Line {_getframe(1).f_lineno}'
    return _print(f"Func {__name__} - {s}", *arg, **kw)

def read_data(args):
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'{data_path} not found')
    if args.dataset == 'hotpotqa':
        with open(data_path,'r',encoding='utf-8') as f:
            data = json.load(f)
    elif args.dataset == 'musique':
        data = []
        with open(data_path,'r',encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    return data

class OllamaUsageHandler(TokenCountingHandler):
    """Collect Ollama-reported token use and server generation durations."""

    def __init__(self):
        super().__init__()
        self.call_metrics = []

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        super().on_event_end(event_type, payload, event_id, **kwargs)
        if event_type != CBEventType.LLM or not payload:
            return
        response = payload.get(EventPayload.RESPONSE)
        raw = getattr(response, 'raw', None)
        raw = raw if isinstance(raw, dict) else {}
        usage = raw.get('usage') or {}
        counts = self.llm_token_counts[-1] if self.llm_token_counts else None
        if counts is None:
            return
        self.call_metrics.append({
            'prompt_tokens': counts.prompt_token_count,
            'completion_tokens': counts.completion_token_count,
            'total_tokens': counts.total_token_count,
            'token_counts_from_ollama': ('prompt_tokens' in usage and
                                         'completion_tokens' in usage),
            'ollama_total_duration_seconds': (
                float(raw['total_duration']) / 1_000_000_000
                if isinstance(raw.get('total_duration'), (int, float)) else None),
        })


def init_model(args):
    usage_handler = OllamaUsageHandler()
    callback_manager = CallbackManager([usage_handler])
    Settings.callback_manager = callback_manager
    Settings.llm = Ollama(model=args.model_name,request_timeout=200,
                          temperature=args.temperature,callback_manager=callback_manager)
    Settings.embed_model = OllamaEmbedding(model_name=args.embed_model_name)
    return usage_handler

def read_kg(args,data):
    if args.dataset=='hotpotqa':
        ents = set()
        for sample in data:
            for ctx in sample['context']:
                ents.add(ctx[0])
        kg_dir = args.kg_dir
        doc2kg = dict()
        print('Loading KGs')
        for ent in tqdm(ents):
            subkg_path = os.path.join(kg_dir,f'{ent.replace("/","_")}.json')
            if os.path.exists(subkg_path):
                with open(subkg_path,'r',encoding='utf-8')as f:
                    subkg = json.load(f)
                    repkg = copy.deepcopy(subkg)
                    if subkg and len(subkg.keys())>0:
                        for seq in subkg.keys():
                            if len(repkg[seq])==0:
                                del repkg[seq]
                        if len(repkg.keys())>0:
                            doc2kg[ent] = repkg
    elif args.dataset=='musique':
        kg_dir = args.kg_dir
        kg_path = os.path.join(kg_dir,'musique_kg_filtered.json')
        with open(kg_path,'r',encoding='utf-8')as f:
            doc2kg = json.load(f)
    print(f'Loaded kg for {len(doc2kg.keys())} entities from {args.dataset}')
    return doc2kg

def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def write_prediction(args,data,prediction):
    result_path = args.result_path
    if args.dataset=='hotpotqa':
        atomic_write(result_path, json.dumps(prediction))
    elif args.dataset=='musique':
        lines = []
        for original in data:
            sample = dict(original)
            sample_id = sample['id']
            if sample_id not in prediction['answer']:
                continue
            sample['predicted_answer'] = prediction['answer'][sample_id]
            sample['predicted_support_idxs'] = prediction['sp'][sample_id]
            sample['predicted_answerable'] = sample['answerable']
            lines.append(json.dumps(sample)+'\n')
        atomic_write(result_path, ''.join(lines))
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    print(f'Prediction written to {result_path}')

def build_postprocessors(args, expansion_pp, filter_pp, naive_pp, encoder=None, cross_encoder=None,
                         native_scores=None, vector_store=None):
    """Keep SEM → CROSS order explicit; CROSS sees only SEM survivors."""
    postprocessors = [expansion_pp]
    if args.method in ('reuse', 'path_reuse'):
        if native_scores is None or vector_store is None:
            raise ValueError('Native score reuse needs the original retrieval score/vector stores')
        postprocessors.append(NativeSemanticScorePostprocessor(store=native_scores, vector_store=vector_store))
    if args.method == 'joint':
        postprocessors.append(JointFilterPostProcessor(
            sem_encoder=encoder, cross_encoder=cross_encoder,
            keep_k=args.joint_keep_k, cross_weight=args.cross_weight))
    if args.method in ('sem', 'sem_cross'):
        postprocessors.append(SemanticFilterPostProcessor(
            encoder=encoder, threshold=args.sem_threshold))
    if args.method in ('cross', 'sem_cross'):
        postprocessors.append(CrossEncoderFilterPostProcessor(
            encoder=cross_encoder if args.method == 'sem_cross' else encoder,
            threshold=args.cross_threshold))
    return postprocessors + [filter_pp, naive_pp]


def process_sample(args,sample,kg,bge_reranker,encoder=None,cross_encoder=None,
                   score_audit=None,usage_audit=None,usage_handler=None):
    sample_started = time.perf_counter()
    if args.dataset=='hotpotqa':
        sample_id = sample['_id']
    elif args.dataset=='musique':
        sample_id = sample['id']
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    sample_question = sample['question']
    sample_answer = sample['answer']

    ents = set()
    subkg = dict()
    doc_chunks = []
    chunks_index = dict()

    if args.dataset=='hotpotqa':
        ctxs = sample['context']
        ents = [ctx[0] for ctx in ctxs]
        for ctx in ctxs:
            ent = ctx[0]
            chunks_index[ent] = {}
            for i in range(len(ctx[1])):
                if (ent in kg) and (str(i) in kg[ent]) and (len(kg[ent][str(i)])>0):
                    if ent not in subkg:
                        subkg[ent] = dict()
                    target_kg = kg[ent][str(i)]
                    for triplet in target_kg:
                        h,r,t = triplet
                        if ngram_overlap(h,ent)>=0.90 or ngram_overlap(ent,h)>=0.90:
                            h = ent
                        if ngram_overlap(t,ent)>=0.90 or ngram_overlap(ent,t)>=0.90:
                            t = ent
                        triplet = (h,r,t)
                    subkg[ent][str(i)] = target_kg
                text = f'{ent}: {ctx[1][i]}'
                doc_chunk = TextNode(text=text,id_=f'{ent}##{str(i)}')
                doc_chunks.append(doc_chunk)
                chunks_index[ent][str(i)] = text
    elif args.dataset=='musique':
        ctxs = sample['paragraphs']
        for ctx in ctxs:
            idx = ctx['idx']
            ent = ctx['title']
            ents.add(ent)
            if ent not in chunks_index:
                chunks_index[ent] = dict()
            seq = ctx['seq']
            text = f'{ent}: {ctx["paragraph_text"]}'
            if (ent in kg) and (str(seq) in kg[ent]) and (len(kg[ent][str(seq)])>0):
                if ent not in subkg:
                    subkg[ent] = dict()
                subkg[ent][f'{str(idx)}##{str(seq)}'] = kg[ent][str(seq)]
            doc_chunk = TextNode(text=text,id_=f'{str(idx)}##{ent}##{str(seq)}')
            doc_chunks.append(doc_chunk)
            chunks_index[ent][f'{str(idx)}##{str(seq)}'] = text

    index = VectorStoreIndex(doc_chunks)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=args.top_k)
    native_scores = NativeScoreStore() if args.method in ('reuse', 'path_reuse') else None
    if native_scores is not None:
        retriever = NativeScoreRetriever(retriever, native_scores)
    qa_rag_template_str = 'Context information is below.\n{context_str}\nThink step by step but give a short factoid answer (as few words as possible) based on the context and your own knowledge.\nQ: Were Scott Derrickson and Ed Wood of the same nationality?\nA: Yes.\nQ: Who was born earlier, Emma Bull or Virginia Woolf?\nA: Adeline Virginia Woolf.\nQ: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nA: 3,677 seated.\nQ: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nA: Chief of Protocol.\n---------------------\nQ: {query_str}\nA: '
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT,text_qa_template=qa_rag_prompt_template)

    expansion_pp = KGRetrievePostProcessor(dataset=args.dataset,ents=ents,doc2kg=subkg,chunks_index=chunks_index)

    filter_pp = GraphFilterPostProcessor(dataset=args.dataset,use_tpt=args.use_tpt,
        topk=(args.joint_keep_k if args.method in ('reuse', 'path_reuse') else
              (args.final_keep_k if args.method == 'baseline' and args.final_keep_k else args.top_k)),
        ents=ents,doc2kg=subkg,chunks_index=chunks_index,reranker=bge_reranker,
        native_scores=native_scores, cross_weight=args.cross_weight if args.method in ('reuse', 'path_reuse') else 0.7,
        path_mode=args.method == 'path_reuse')
    naive_pp = NaivePostprocessor(dataset=args.dataset)

    postprocessors = build_postprocessors(
        args, expansion_pp, filter_pp, naive_pp, encoder, cross_encoder, native_scores,
        index.vector_store)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=postprocessors,
    )

    llm_call_start = len(usage_handler.call_metrics) if usage_handler is not None else 0
    query_started = time.perf_counter()
    try:
        response = query_engine.query(sample_question)
        prediction = response.response
        if args.dataset=='hotpotqa':
            sps = [[source_node.node.id_.split('##')[0],int(source_node.node.id_.split('##')[1])] for source_node in response.source_nodes]
            sps = [[ent,seq]for ent,seq in sps if (seq>=0)]
        elif args.dataset=='musique':
            sps = [int(source_node.node.id_.split('##')[0]) for source_node in response.source_nodes]
            sps = [idx for idx in sps if (idx>=0)]
    except Exception as e:
        raise RuntimeError(f'Sample {sample_id}: {e}') from e
    finally:
        query_latency = time.perf_counter() - query_started
        if usage_audit is not None:
            calls = (usage_handler.call_metrics[llm_call_start:]
                     if usage_handler is not None else [])
            ollama_durations = [call['ollama_total_duration_seconds'] for call in calls
                                if call['ollama_total_duration_seconds'] is not None]
            usage_audit.update(
                llm_call_count=len(calls),
                prompt_tokens=sum(call['prompt_tokens'] for call in calls),
                completion_tokens=sum(call['completion_tokens'] for call in calls),
                total_tokens=sum(call['total_tokens'] for call in calls),
                token_counts_from_ollama=bool(calls) and all(
                    call['token_counts_from_ollama'] for call in calls),
                ollama_generation_seconds=sum(ollama_durations),
                ollama_timed_call_count=len(ollama_durations),
                query_latency_seconds=query_latency)
    if not isinstance(prediction, str) or not prediction.strip():
        raise ValueError(f'Sample {sample_id}: empty answer')
    if native_scores is not None and score_audit is not None:
        score_audit.update(seed_scores_reused=native_scores.seed_scores_reused,
                           expanded_vectors_reused=native_scores.expanded_vectors_reused,
                           semantic_scores=dict(native_scores.scores), selection=filter_pp.reuse_audit)
    if usage_audit is not None:
        usage_audit['sample_pipeline_seconds'] = time.perf_counter() - sample_started
    return sample_id,prediction,sps

def run_config(args, data, kg):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    source_hash = hashlib.sha256()
    for path in [Path(__file__), *sorted((Path(__file__).parent / 'util').glob('*.py'))]:
        source_hash.update(path.read_bytes())
    return {
        'dataset': args.dataset, 'data_hash': digest(data), 'kg_hash': digest(kg),
        'source_hash': source_hash.hexdigest(),
        'model_name': args.model_name, 'embed_model_name': args.embed_model_name,
        'temperature': args.temperature,
        'reranker': args.reranker, 'reranker_device': args.reranker_device,
        'reranker_fp16': args.reranker_fp16, 'top_k': args.top_k,
        'use_tpt': args.use_tpt,
        'method': args.method,
        'sem_model': args.sem_model if args.method in ('sem', 'sem_cross', 'joint') else None,
        'sem_threshold': args.sem_threshold if args.method in ('sem', 'sem_cross') else None,
        'cross_model': args.cross_model if args.method in ('cross', 'sem_cross', 'joint') else None,
        'cross_threshold': args.cross_threshold if args.method in ('cross', 'sem_cross') else None,
        'joint_keep_k': args.joint_keep_k if args.method in ('joint', 'reuse', 'path_reuse') else None,
        'cross_weight': args.cross_weight if args.method in ('joint', 'reuse', 'path_reuse') else None,
        'final_keep_k': args.final_keep_k if args.method == 'baseline' else None,
        'native_reranker_identity': getattr(args, 'native_reranker_identity', None),
    }


def load_checkpoint(args, data, kg):
    if not data:
        raise ValueError('Dataset is empty')
    key = '_id' if args.dataset == 'hotpotqa' else 'id'
    ids = [sample[key] for sample in data]
    if len(set(ids)) != len(ids):
        raise ValueError('Dataset contains duplicate question IDs')
    path = Path(str(args.result_path) + '.checkpoint.json')
    config = run_config(args, data, kg)
    if args.resume:
        if not path.exists():
            raise ValueError(f'No checkpoint at {path}; use a new output path for a fresh run')
        state = json.loads(path.read_text())
        if state['config'] != config:
            raise ValueError('Checkpoint settings, source, dataset, or graphs differ; use a new output path')
        prediction = state['prediction']
        if (set(prediction['answer']) != set(prediction['sp']) or
                not set(prediction['answer']).issubset(ids) or
                not set(state['errors']).issubset(ids) or
                set(state['errors']) & set(prediction['answer'])):
            raise ValueError('Checkpoint contains inconsistent question IDs')
        return state
    if path.exists() or Path(args.result_path).exists():
        raise FileExistsError('Output already exists; use --resume or a new result path')
    return {'config': config, 'prediction': {'answer': {}, 'sp': {}}, 'errors': {}}


def save_checkpoint(args, data, state):
    # The sidecar is authoritative if interrupted between these two atomic writes.
    atomic_write(str(args.result_path) + '.checkpoint.json', json.dumps(state))
    write_prediction(args, data, state['prediction'])


def write_usage_report(args, state):
    questions = state.get('usage', {})

    def summarize(key):
        values = [float(row[key]) for row in questions.values() if key in row]
        return {'mean': statistics.mean(values) if values else None,
                'std': statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)}

    report = {
        'scope': 'question-answering inference; KG extraction is excluded',
        'model_name': args.model_name,
        'question_count': len(questions),
        'all_token_counts_from_ollama': bool(questions) and all(
            row.get('token_counts_from_ollama', False) for row in questions.values()),
        'totals': {
            key: sum(int(row.get(key, 0)) for row in questions.values())
            for key in ['llm_call_count', 'prompt_tokens', 'completion_tokens',
                        'total_tokens', 'ollama_timed_call_count']
        },
        'per_question_mean_std': {
            key: summarize(key) for key in [
                'prompt_tokens', 'completion_tokens', 'total_tokens',
                'query_latency_seconds', 'sample_pipeline_seconds',
                'ollama_generation_seconds']
        },
        'questions': questions,
    }
    atomic_write(Path(args.result_path).with_name('usage.json'), json.dumps(report))


def kgrag_distractor_predict(args,data,kg,state,shared_reranker=None):
    encoder = None
    cross_encoder = None
    usage_handler = None
    prediction = state['prediction']
    pending = [sample for sample in data
               if sample['_id' if args.dataset == 'hotpotqa' else 'id'] not in prediction['answer']]
    print(f'Completed: {len(prediction["answer"])}; pending: {len(pending)}')
    save_checkpoint(args, data, state)
    if pending:
        usage_handler = init_model(args)
        print(f'Reranker device: {args.reranker_device}; cached/lazy: {shared_reranker is not None}')
        reranker = shared_reranker if shared_reranker is not None else FlagReranker(
            model_name_or_path=args.reranker, devices=[args.reranker_device], use_fp16=args.reranker_fp16)
        if args.method in ('sem', 'sem_cross', 'joint'):
            encoder = SentenceTransformer(
                args.sem_model,
                device='cpu',
            )
        if args.method == 'cross':
            encoder = CrossEncoder(
                args.cross_model,
                device='cpu',
            )
        if args.method in ('sem_cross', 'joint'):
            cross_encoder = CrossEncoder(args.cross_model, device='cpu')
    for sample in tqdm(pending):
        sample_id = sample['_id' if args.dataset == 'hotpotqa' else 'id']
        try:
            audit = {}
            usage_audit = {}
            if args.method in ('reuse', 'path_reuse'):
                _,sample_prediction,sample_sps = process_sample(args,sample,kg,reranker,encoder,cross_encoder,
                                                               score_audit=audit, usage_audit=usage_audit,
                                                               usage_handler=usage_handler)
                state.setdefault('score_reuse', {})[sample_id] = audit
            else:
                _,sample_prediction,sample_sps = process_sample(args,sample,kg,reranker,encoder,cross_encoder,
                                                               score_audit=audit, usage_audit=usage_audit,
                                                               usage_handler=usage_handler)
            state.setdefault('usage', {})[sample_id] = usage_audit
            prediction['answer'][sample_id] = sample_prediction
            prediction['sp'][sample_id] = sample_sps
            state['errors'].pop(sample_id, None)
        except Exception as error:
            logger.exception('Question %s failed', sample_id)
            state['errors'][sample_id] = str(error)
            if usage_audit:
                state.setdefault('usage', {})[sample_id] = usage_audit
        save_checkpoint(args, data, state)
        write_usage_report(args, state)
    if state['errors']:
        raise RuntimeError(f'{len(state["errors"])} questions failed; inspect the checkpoint and rerun with --resume')
    print(f'Average number of supporting facts: {sum(map(len, prediction["sp"].values()))/len(data)}')
    return prediction

def main(args):
    data = read_data(args)
    kg = read_kg(args,data)
    shared = None
    args.native_reranker_identity = None
    cache_dir = args.score_cache_dir
    if args.method in ('reuse', 'path_reuse') and not cache_dir:
        cache_dir = str(Path(__file__).resolve().parents[1] / 'output/score_cache')
    if cache_dir:
        checkpoint, identity = pinned_reranker(args.reranker)
        scoring = {'devices': [args.reranker_device], 'use_fp16': args.reranker_fp16,
                   'batch_size': 128, 'max_length': 512, 'normalize': False}
        shared = CachedReranker(Path(cache_dir) / 'bge.sqlite',
            lambda: FlagReranker(model_name_or_path=checkpoint, **scoring), identity, scoring)
        args.native_reranker_identity = {'model': identity, 'scoring': scoring, 'namespace': shared.namespace}
    state, succeeded = None, False
    try:
        state = load_checkpoint(args, data, kg)
        kgrag_distractor_predict(args,data,kg,state,shared)
        succeeded = True
    finally:
        try:
            if state is not None:
                write_usage_report(args, state)
        finally:
            if shared is not None:
                try:
                    if state is not None:
                        report_path = Path(args.result_path).parent / 'score_reuse.json'
                        report = json.loads(report_path.read_text()) if report_path.exists() else {'attempts': []}
                        report['attempts'].append({'completed': succeeded, 'bge': dict(shared.stats),
                                                   'model_loaded': shared.model is not None,
                                                   'namespace': shared.namespace})
                        report['questions'] = state.get('score_reuse', {})
                        atomic_write(report_path, json.dumps(report))
                finally:
                    shared.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # hotpotqa distractor
    parser.add_argument('--dataset',type=str,default='hotpotqa',help='Dataset name')
    parser.add_argument('--data_path',type=str,default='../data/hotpotqa/hotpot_dev_distractor_v1.json',help='Path to the data file')
    parser.add_argument('--kg_dir',type=str,default='../data/hotpotqa/kgs/extract_subkgs',help='Directory of the KGs')
    parser.add_argument('--use_tpt',type=bool,default=False,help='Whether to use triplet representation')
    parser.add_argument('--result_path',type=str,default='../output/hotpot/hotpot_dev_distractor_v1_kgrag.json',help='Path to the result file')

    # # pu-hotpotqa distractor
    # parser.add_argument('--dataset',type=str,default='hotpotqa',help='Dataset name')
    # parser.add_argument('--data_path',type=str,default='../data/pu-hotpotqa/hotpot_dev_distractor_v1.json',help='Path to the data file')
    # parser.add_argument('--kg_dir',type=str,default='../data/pu-hotpotqa/kgs/extract_subkgs',help='Directory of the KGs')
    # parser.add_argument('--use_tpt',type=bool,default=False,help='Whether to use triplet representation')
    # parser.add_argument('--result_path',type=str,default='../output/pu-hotpot/pu-hotpot_dev_distractor_v1_kgrag.json',help='Path to the result file')

    # # musique distractor
    # parser.add_argument('--dataset',type=str,default='musique',help='Dataset name')
    # parser.add_argument('--data_path',type=str,default='../data/MuSiQue/musique_ans_v1.0_dev_mapped.jsonl',help='Path to the data file')
    # parser.add_argument('--kg_dir',type=str,default='../data/MuSiQue/kgs/extract_subkgs',help='Directory of the KGs')
    # parser.add_argument('--use_tpt',type=bool,default=True,help='Whether to use triplet representation')
    # parser.add_argument('--result_path',type=str,default='../output/musique/musique_dev_kgrag.jsonl',help='Path to the result file')

    # # trivia
    # parser.add_argument('--dataset',type=str,default='hotpotqa',help='Dataset name')
    # parser.add_argument('--data_path',type=str,default='../data/trivia_qa/trivia.json',help='Path to the data file')
    # parser.add_argument('--kg_dir',type=str,default='../data/trivia_qa/kgs/extracted_subkgs',help='Directory of the KGs')
    # parser.add_argument('--use_tpt',type=bool,default=True,help='Whether to use triplet representation')
    # parser.add_argument('--result_path',type=str,default='../output/trivia_qa/trivia_kgrag.json',help='Path to the result file')

    parser.add_argument('--embed_model_name',type=str,default='mxbai-embed-large',help='Ollama embedding model name for indexing')
    parser.add_argument('--model_name',type=str,default='llama3:8b',help='Ollama model name')
    parser.add_argument('--reranker',type=str,default='../model/bge-reranker-large',help='Path of the reranker model')
    parser.add_argument('--top_k',type=int,default=10,help='Top k similar documents')
    parser.add_argument('--temperature',type=float,default=0.0,help='Ollama generation temperature (0 for repeatable comparisons)')
    parser.add_argument('--reranker_device', default='cpu', help='Reranker device, e.g. cpu or cuda:0')
    parser.add_argument('--reranker_fp16', action='store_true', help='Enable reranker half precision on GPU')
    parser.add_argument('--resume', action='store_true', help='Resume matching checkpoint and retry failures')

    # SEM or CROSS
    parser.add_argument('--method', choices=['baseline', 'sem', 'cross', 'sem_cross', 'joint', 'reuse', 'path_reuse'], default='baseline')
    parser.add_argument('--score_cache_dir', default=None, help='Optional exact-input BGE cache; reuse mode enables one by default')
    parser.add_argument('--joint_keep_k', type=int, default=10)
    parser.add_argument('--final_keep_k', type=int, default=None,
                        help='Final evidence sentence cap for a baseline run; defaults to top_k')
    parser.add_argument('--cross_weight', type=float, default=0.7)

    # SEM
    parser.add_argument('--sem_model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--sem_threshold', type=float, default=0.5)

    # CROSS
    parser.add_argument('--cross_model', default='cross-encoder/ms-marco-MiniLM-L6-v2')
    parser.add_argument('--cross_threshold', type=float, default=0.5)

    args = parser.parse_args()

    main(args)
