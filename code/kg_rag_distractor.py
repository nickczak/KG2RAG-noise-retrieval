import os
import copy
import ujson as json
import argparse
import hashlib
import tempfile
from pathlib import Path
from tqdm import tqdm
from FlagEmbedding import FlagReranker
from llama_index.core import Settings,VectorStoreIndex,PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from util.kg_post_processor import NaivePostprocessor,KGRetrievePostProcessor,ngram_overlap, GraphFilterPostProcessor, SemanticFilterPostProcessor, CrossEncoderFilterPostProcessor
from util.kg_response_synthesizer import get_response_synthesizer
from sentence_transformers import SentenceTransformer, CrossEncoder

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

def init_model(args):
    Settings.llm = Ollama(model=args.model_name,request_timeout=200)
    Settings.embed_model = OllamaEmbedding(model_name=args.embed_model_name)

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

def process_sample(args,sample,kg,bge_reranker,encoder=None):
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
    qa_rag_template_str = 'Context information is below.\n{context_str}\nThink step by step but give a short factoid answer (as few words as possible) based on the context and your own knowledge.\nQ: Were Scott Derrickson and Ed Wood of the same nationality?\nA: Yes.\nQ: Who was born earlier, Emma Bull or Virginia Woolf?\nA: Adeline Virginia Woolf.\nQ: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nA: 3,677 seated.\nQ: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nA: Chief of Protocol.\n---------------------\nQ: {query_str}\nA: '
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT,text_qa_template=qa_rag_prompt_template)

    expansion_pp = KGRetrievePostProcessor(dataset=args.dataset,ents=ents,doc2kg=subkg,chunks_index=chunks_index)

    filter_pp = GraphFilterPostProcessor(dataset=args.dataset,use_tpt=args.use_tpt,topk=args.top_k,ents=ents,doc2kg=subkg,chunks_index=chunks_index,reranker=bge_reranker)
    naive_pp = NaivePostprocessor(dataset=args.dataset)

    # SEM and CROSS filter here
    postprocessors = [expansion_pp]
    if args.method == 'sem':
        sem_pp = SemanticFilterPostProcessor(
            encoder=encoder,
            threshold=args.sem_threshold,
        )
        postprocessors.append(sem_pp)
    if args.method == 'cross':
        cross_pp = CrossEncoderFilterPostProcessor(
            encoder=encoder,
            threshold=args.cross_threshold,
        )
        postprocessors.append(cross_pp)
    postprocessors.extend([filter_pp, naive_pp])
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=postprocessors,
    )

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
    if not isinstance(prediction, str) or not prediction.strip():
        raise ValueError(f'Sample {sample_id}: empty answer')
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
        'reranker': args.reranker, 'reranker_device': args.reranker_device,
        'reranker_fp16': args.reranker_fp16, 'top_k': args.top_k,
        'use_tpt': args.use_tpt,
        'method': args.method,
        'sem_model': args.sem_model if args.method == 'sem' else None,
        'sem_threshold': args.sem_threshold if args.method == 'sem' else None,
        'cross_model': args.cross_model if args.method == 'cross' else None,
        'cross_threshold': args.cross_threshold if args.method == 'cross' else None,
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


def kgrag_distractor_predict(args,data,kg,state):
    encoder = None
    prediction = state['prediction']
    pending = [sample for sample in data
               if sample['_id' if args.dataset == 'hotpotqa' else 'id'] not in prediction['answer']]
    print(f'Completed: {len(prediction["answer"])}; pending: {len(pending)}')
    save_checkpoint(args, data, state)
    if pending:
        init_model(args)
        print(f'Loading reranker once on {args.reranker_device}')
        reranker = FlagReranker(model_name_or_path=args.reranker,
                               devices=[args.reranker_device], use_fp16=args.reranker_fp16)
        if args.method == 'sem':
            encoder = SentenceTransformer(
                args.sem_model,
                device='cpu',
            )
        if args.method == 'cross':
            encoder = CrossEncoder(
                args.cross_model,
                device='cpu',
            )
    for sample in tqdm(pending):
        sample_id = sample['_id' if args.dataset == 'hotpotqa' else 'id']
        try:
            _,sample_prediction,sample_sps = process_sample(args,sample,kg,reranker,encoder)
            prediction['answer'][sample_id] = sample_prediction
            prediction['sp'][sample_id] = sample_sps
            state['errors'].pop(sample_id, None)
        except Exception as error:
            logger.exception('Question %s failed', sample_id)
            state['errors'][sample_id] = str(error)
        save_checkpoint(args, data, state)
    if state['errors']:
        raise RuntimeError(f'{len(state["errors"])} questions failed; inspect the checkpoint and rerun with --resume')
    print(f'Average number of supporting facts: {sum(map(len, prediction["sp"].values()))/len(data)}')
    return prediction

def main(args):
    data = read_data(args)
    kg = read_kg(args,data)
    state = load_checkpoint(args, data, kg)
    kgrag_distractor_predict(args,data,kg,state)

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
    parser.add_argument('--reranker_device', default='cpu', help='Reranker device, e.g. cpu or cuda:0')
    parser.add_argument('--reranker_fp16', action='store_true', help='Enable reranker half precision on GPU')
    parser.add_argument('--resume', action='store_true', help='Resume matching checkpoint and retry failures')

    # SEM or CROSS
    parser.add_argument('--method', choices=['baseline', 'sem', 'cross'], default='baseline')

    # SEM
    parser.add_argument('--sem_model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--sem_threshold', type=float, default=0.5)

    # CROSS
    parser.add_argument('--cross_model', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--cross_threshold', type=float, default=0.5)

    args = parser.parse_args()

    main(args)
