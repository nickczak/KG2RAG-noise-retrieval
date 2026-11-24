import random
import re
import time
import networkx as nx
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from numpy import argmax

from typing import Any,List,Dict,Optional,Set
from FlagEmbedding import FlagReranker
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation import get_dispatcher
dispatcher = get_dispatcher(__name__)

from builtins import print as _print
from sys import _getframe
def print(*arg, **kw):
    s = f'Line {_getframe(1).f_lineno}'
    return _print(f"Func {__name__} - {s}", *arg, **kw)

import string
def ngram_overlap(span,sent,n=3):
    while (len(span)<n) or (len(sent)<n):
        n -= 1
    if n<=0:
        return 0.0
    span = span.lower()
    sent = sent.lower()
    span_tokens = [token for token in span.split() if token not in string.punctuation]
    span_tokens = ''.join(span_tokens)
    sent_tokens = [token for token in sent.split() if token not in string.punctuation]
    sent_tokens = ''.join(sent_tokens)
    span_tokens = set([span_tokens[i:i+n] for i in range(len(span_tokens)-n+1)])
    sent_tokens = set([sent_tokens[i:i+n] for i in range(len(sent_tokens)-n+1)])
    overlap = span_tokens.intersection(sent_tokens)
    return float((len(overlap)+0.01)/(len(span_tokens)+0.01))

class NaivePostprocessor(BaseNodePostprocessor):
    """Naive Node Postprocessor."""

    dataset: str = Field

    @classmethod
    def class_name(cls) -> str:
        return "NaivePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes"""
        entity_order = {}
        sorted_nodes = []
        for i,node in enumerate(nodes):
            node_id = node.node.id_
            if self.dataset=='hotpotqa':
                ent,seq_str = node_id.split('##')
                idx_seq_str = seq_str
                ctx_seq = int(seq_str)
            elif self.dataset=='musique':
                idx_str,ent,seq_str = node_id.split('##')
                idx_seq_str = f'{idx_str}##{seq_str}'
                ctx_seq = i
            if ent not in entity_order:
                entity_order[ent] = len(entity_order)
            sorted_nodes.append((ent,ctx_seq,node))
        sorted_nodes.sort(key=lambda x:(entity_order[x[0]],x[1]))
        sorted_nodes = [node for _,_,node in sorted_nodes]

        prev_ent = ''
        for i in range(0,len(sorted_nodes)):
            if self.dataset=='hotpotqa':
                temp_ent = sorted_nodes[i].node.id_.split('##')[0]
            elif self.dataset=='musique':
                temp_ent = sorted_nodes[i].node.id_.split('##')[1]
            if (prev_ent == temp_ent):
                sorted_nodes[i].node.text = sorted_nodes[i].node.text[len(temp_ent+': '):]
            if i<len(sorted_nodes)-1:
                if self.dataset=='hotpotqa':
                    next_ent = sorted_nodes[i+1].node.id_.split('##')[0]
                elif self.dataset=='musique':
                    next_ent = sorted_nodes[i+1].node.id_.split('##')[1]
                if next_ent!=temp_ent:
                    sorted_nodes[i].node.text += '\n'
            prev_ent = temp_ent
     
        return sorted_nodes
    
class KGRetrievePostProcessor(BaseNodePostprocessor):

    """KnowledgeGraph-based Node processor."""

    dataset: str = Field
    ents: Set[str] = Field
    doc2kg: Dict[str,Dict[str,List[List[str]]]] = Field
    chunks_index: Dict[str,Dict[str,str]] = Field

    @classmethod
    def class_name(cls) -> str:
        return "KGRetrievePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes"""
        top_k = len(nodes)

        retrieved_ids = set()
        retrieved_ents = set()

        related_ents = set()
        highly_related_ents = set()

        textid2score = dict()
        ent_count = dict()
        ent_score = dict()

        for i in range(len(nodes)):
            node = nodes[i]
            node_id = node.node.id_
            retrieved_ids.add(node_id)
            textid2score[node_id] = node.score
            if self.dataset=='hotpotqa':
                entity,seq_str = node_id.split('##')
            elif self.dataset=='musique':
                idx_str,entity,seq_str = node_id.split('##')

            if (i<(top_k//2)) and (entity in retrieved_ents):
                highly_related_ents.add(entity)
            retrieved_ents.add(entity)

            if entity not in ent_count:
                ent_count[entity] = 0
                ent_score[entity] = 0.0
            ent_count[entity] += 1
            ent_score[entity] += node.score

        sorted_ents = sorted(ent_count.keys(),key=lambda x:(ent_score[x]/ent_count[x]),reverse=True)
        for i in range(min(2,len(sorted_ents))):
            highly_related_ents.add(sorted_ents[i])

        additional_ents = set()
        for node in nodes:
            node_id = node.node.id_
            if self.dataset=='hotpotqa':
                entity,seq_str = node_id.split('##')
                idx_seq_str = seq_str
            elif self.dataset=='musique':
                idx_str,entity,seq_str = node_id.split('##')
                idx_seq_str = f'{idx_str}##{seq_str}'
            if (entity not in self.doc2kg) or (idx_seq_str not in self.doc2kg[entity]):
                continue
            for triplet in self.doc2kg[entity][idx_seq_str]:
                h,r,t = triplet
                triplet = [h,r,t]
                if (h in self.ents) and (h not in retrieved_ents):
                    additional_ents.add(h)
                    if h not in ent_count:
                        ent_count[h] = 0
                        ent_score[h] = 0.0
                    ent_count[h] += 1
                    ent_score[h] += node.score
                if (t in self.ents) and (t not in retrieved_ents):
                    additional_ents.add(t)
                    if t not in ent_count:
                        ent_count[t] = 0
                        ent_score[t] = 0.0
                    ent_count[t] += 1
                    ent_score[t] += node.score

        additional_ents = additional_ents.union(retrieved_ents)
        hops = 1
        for hop in range(hops):
            related_ents = related_ents.union(additional_ents)
            temp_ents = set(additional_ents)
            additional_ents = set()
            for ent in temp_ents:

                if (ent not in self.doc2kg) or (len(self.doc2kg[ent])==0):
                    continue
                for idx_seq_str in self.doc2kg[ent]:
                    if len(self.doc2kg[ent][idx_seq_str])==0:
                        continue
                    if self.dataset=='hotpotqa':
                        ctx_id = f'{ent}##{idx_seq_str}'
                    elif self.dataset=='musique':
                        idx_str,seq_str = idx_seq_str.split('##')
                        ctx_id = f'{idx_str}##{ent}##{seq_str}'
                    if ctx_id in retrieved_ids:
                        continue
                    
                    for triplet in self.doc2kg[ent][idx_seq_str]:
                        h,r,t = triplet

                        if (h in self.ents) and (h not in related_ents):
                            additional_ents.add(h)
                            if h not in ent_count:
                                ent_count[h] = 0
                                ent_score[h] = 0.0
                            ent_count[h] += 1
                            ent_score[h] += float((ent_score[ent])/(ent_count[ent]))
                        if (t in self.ents) and (t not in related_ents):
                            additional_ents.add(t)
                            if t not in ent_count:
                                ent_count[t] = 0
                                ent_score[t] = 0.0
                            ent_count[t] += 1
                            ent_score[t] += float((ent_score[ent])/(ent_count[ent]))

        related_ents = related_ents.union(additional_ents)
        additional_ids = set()

        avg_score = float(sum([node.score for node in nodes])/len(nodes))
        retrieved_ents = retrieved_ents-highly_related_ents
        for ent in (related_ents-retrieved_ents):
            if (ent not in self.chunks_index) or (len(self.chunks_index[ent])==0) or (ent not in self.doc2kg):
                continue
            for idx_seq_str in self.chunks_index[ent]:
                if self.dataset=='hotpotqa':
                    ctx_id = f'{ent}##{idx_seq_str}'
                elif self.dataset=='musique':
                    idx_str,seq_str = idx_seq_str.split('##')
                    ctx_id = f'{idx_str}##{ent}##{seq_str}'
                if ctx_id in retrieved_ids:
                    continue
                additional_ids.add(ctx_id)
                textid2score[ctx_id] = 0.0
                if ent in ent_score:
                    textid2score[ctx_id] += (ent_score[ent]+avg_score)/(ent_count[ent]+1)

        added_nodes = []
        for ctx_id in additional_ids:
            if self.dataset=='hotpotqa':
                ent,seq_str = ctx_id.split('##')
                idx_seq_str = seq_str
            elif self.dataset=='musique':
                idx_str,ent,seq_str = ctx_id.split('##')
                idx_seq_str = f'{idx_str}##{seq_str}'
            if ent in self.chunks_index:
                if idx_seq_str in self.chunks_index[ent]:
                    ctx_text = self.chunks_index[ent][idx_seq_str]
                    node = TextNode(id_=ctx_id,text=ctx_text)
                    node = NodeWithScore(node=node,score=textid2score[ctx_id])
                    added_nodes.append(node)
        added_nodes = sorted(added_nodes,key=lambda x:x.score,reverse=True)

        nodes = nodes+added_nodes
        return nodes

class GraphFilterPostProcessor(BaseNodePostprocessor):

    """KnowledgeGraph-based Node processor."""
    dataset: str = Field
    topk: int = Field
    use_tpt: bool = Field
    ents: Set[str] = Field
    doc2kg: Dict[str,Dict[str,List[List[str]]]] = Field
    chunks_index: Dict[str,Dict[str,str]] = Field
    reranker: Any = Field
    native_scores: Any = None
    path_mode: bool = False
    cross_weight: float = Field(default=0.7, ge=0, le=1)
    reuse_audit: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def class_name(cls) -> str:
        return "GraphFilterPostprocessor"

    def _native_rank(self, source_lists, texts, question):
        """Fuse cached native cosine and BGE ranks at the same candidate unit."""
        cross = np.asarray(self.reranker.compute_score([(question, text) for text in texts]),
                           dtype=float).reshape(-1)
        if len(cross) != len(texts) or not np.all(np.isfinite(cross)):
            raise ValueError('Invalid native BGE group scores')
        try:
            semantic = np.array([np.mean([self.native_scores.scores[source]
                                         for source in dict.fromkeys(sources)])
                                 for sources in source_lists])
        except KeyError as error:
            raise ValueError(f'Missing native semantic score for {error}') from error
        if not np.all(np.isfinite(semantic)):
            raise ValueError('Invalid native group semantic similarities')
        fused = self.cross_weight * _rank_percentiles(cross) + (1-self.cross_weight) * _rank_percentiles(semantic)
        traces = [{'source_ids': list(sources), 'semantic_mean': float(sem),
                   'bge_score': float(bge), 'fused_rank': float(rank)}
                  for sources, sem, bge, rank in zip(source_lists, semantic, cross, fused)]
        return semantic, cross, fused, traces

    def _select_native(self, nodes, query_bundle, source_lists, texts):
        """Use KG paths for the active method or preserve complete KG2RAG groups."""
        if not nodes:
            return []
        if self.path_mode:
            return self._select_paths(nodes, query_bundle)
        groups = [(sources, text) for sources, text in zip(source_lists, texts) if sources]
        selected, chosen, group_trace = set(), [], []
        if groups:
            sources, group_texts = map(list, zip(*groups))
            sem, cross, fused, group_trace = self._native_rank(sources, group_texts, query_bundle.query_str)
            order = sorted(range(len(groups)), key=lambda i: (-fused[i], -cross[i], -sem[i], i))
            anchor = max(range(len(groups)), key=lambda i: (cross[i], sem[i], -i))
            if len(set(sources[anchor])) <= self.topk:
                order = [anchor] + [i for i in order if i != anchor]
            for i in order:
                combined = selected | set(sources[i])
                if len(combined) > self.topk:
                    continue
                selected = combined
                chosen.append(i)
            for i, trace in enumerate(group_trace):
                trace['selected_group'] = i in chosen
        self.reuse_audit = {'selection_unit': 'complete_groups', 'group_scores': group_trace,
                            'sentence_scores': [], 'kept_ids': []}
        if not selected:
            sources = [[node.node.id_] for node in nodes]
            sem, cross, _, trace = self._native_rank(sources, [node.node.text for node in nodes], query_bundle.query_str)
            kept = select_joint_chunks(nodes, sem, cross, self.topk, self.cross_weight)
            selected = {node.node.id_ for node in kept}
            for row in trace:
                row['selected'] = row['source_ids'][0] in selected
            self.reuse_audit.update(selection_unit='sentence_fallback', sentence_scores=trace)
        self.reuse_audit['kept_ids'] = [node.node.id_ for node in nodes if node.node.id_ in selected]
        return [node for node in nodes if node.node.id_ in selected]

    def _select_paths(self, nodes, query_bundle):
        """Rank candidate endpoints and retain their shortest paths from retrieval seeds."""
        node_by_id = {node.node.id_: node for node in nodes}
        ids = list(node_by_id)
        graph = nx.Graph()
        entities_by_id = {}
        for node in nodes:
            parts = node.node.id_.split('##')
            title, sentence = parts[-2], parts[-1]
            triples = self.doc2kg.get(title, {}).get(sentence, [])
            entities = {str(value).strip().casefold() for triple in triples
                        for value in (triple[0], triple[2])
                        if len(triple) >= 3 and str(value).strip()}
            entities_by_id[node.node.id_] = entities
            graph.add_node(node.node.id_)
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                shared = entities_by_id[left] & entities_by_id[right]
                if shared:
                    graph.add_edge(left, right, bridges=sorted(shared))

        seeds = sorted(self.native_scores.seed_ids & set(node_by_id))
        if not seeds:
            raise ValueError('Path selection requires at least one original retrieval seed')
        query = query_bundle.query_str
        cross = np.asarray(self.reranker.compute_score(
            [(query, node_by_id[node_id].node.text) for node_id in ids]), dtype=float).reshape(-1)
        semantic = np.asarray([self.native_scores.scores[node_id] for node_id in ids], dtype=float)
        fused = (self.cross_weight * _rank_percentiles(cross) +
                 (1-self.cross_weight) * _rank_percentiles(semantic))
        rank = sorted(range(len(ids)), key=lambda i: (-fused[i], -cross[i], -semantic[i], ids[i]))

        chosen, paths = set(), []
        for i in rank:
            endpoint = ids[i]
            options = []
            for seed in seeds:
                try:
                    path = nx.shortest_path(graph, seed, endpoint)
                    options.append((len(path) - 1, tuple(path)))
                except nx.NetworkXNoPath:
                    continue
            if not options:
                continue
            hops, path = min(options, key=lambda item: (item[0], item[1]))
            if len(chosen | set(path)) <= self.topk:
                chosen.update(path)
                paths.append({'endpoint': endpoint, 'path': list(path), 'hops': hops,
                              'bridges': [graph[a][b]['bridges'] for a, b in zip(path, path[1:])],
                              'semantic_score': float(semantic[i]), 'bge_score': float(cross[i]),
                              'fused_rank': float(fused[i])})
        if not chosen:
            chosen.add(seeds[0])
        self.reuse_audit = {'selection_unit': 'shortest_seed_paths', 'seed_ids': seeds,
                            'paths': paths, 'path_selected_ids': [node_id for node_id in ids if node_id in chosen],
                            'candidate_count': len(nodes), 'edge_count': graph.number_of_edges(),
                            'kept_ids': [node_id for node_id in ids if node_id in chosen]}
        return [node for node in nodes if node.node.id_ in chosen]

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes"""
        ents = set()
        rels = set()

        g = nx.MultiGraph()

        for node in nodes:
            if self.dataset=='hotpotqa':
                ent,seq_str = node.node.id_.split('##')
            elif self.dataset=='musique':
                idx_str,ent,seq_str = node.node.id_.split('##')
            ents.add(ent)

        for node in nodes:
            if self.dataset=='hotpotqa':
                ent,seq_str = node.node.id_.split('##')
                idx_seq_str = seq_str
            elif self.dataset=='musique':
                idx_str,ent,seq_str = node.node.id_.split('##')
                idx_seq_str = f'{idx_str}##{seq_str}'
            if (ent not in self.doc2kg) or (idx_seq_str not in self.doc2kg[ent]) or (len(self.doc2kg[ent][idx_seq_str])==0):
                continue
            for triplet in self.doc2kg[ent][idx_seq_str]:
                h,r,t = triplet
                h = h.strip()
                r = r.strip()
                t = t.strip()
                triplet = [h,r,t]
                ents.add(h)
                ents.add(t)
                rels.add(r)
                g.add_edge(h,t,rel=r,source=node.node.id_,weight=node.score)

        mentioned_ents = set()
        mentioned_rels = set()

        for ent in ents:
            overlap_score = ngram_overlap(ent,query_bundle.query_str)
            if overlap_score>=0.90:
                mentioned_ents.add(ent)
        for rel in rels:
            overlap_score = ngram_overlap(rel,query_bundle.query_str)
            if overlap_score>=0.90:
                mentioned_rels.add(rel)

        for node in nodes:
            if self.dataset=='hotpotqa':
                ent,seq_str = node.node.id_.split('##')
                idx_seq_str = seq_str
            elif self.dataset=='musique':
                idx_str,ent,seq_str = node.node.id_.split('##')
                idx_seq_str = f'{idx_str}##{seq_str}'
            if (ent not in self.doc2kg) or (idx_seq_str not in self.doc2kg[ent]) or (len(self.doc2kg[ent][idx_seq_str])==0):
                continue
            for triplet in self.doc2kg[ent][idx_seq_str]:
                h,r,t = triplet
                triplet = [h,r,t]
                if (h in mentioned_ents) and (r in mentioned_rels) and (not t in mentioned_ents):
                    mentioned_ents.add(t)
                if (t in mentioned_ents) and (r in mentioned_rels) and (not h in mentioned_ents):
                    mentioned_ents.add(h)

        mentioned_ents_list = list(mentioned_ents)
        for i in range(len(mentioned_ents_list)):
            for j in range(i+1,len(mentioned_ents_list)):
                if (g.has_edge(mentioned_ents_list[i],mentioned_ents_list[j])) or (g.has_edge(mentioned_ents_list[j],mentioned_ents_list[i])):
                    continue
                g.add_edge(mentioned_ents_list[i],mentioned_ents_list[j],rel='cooccurrence',source='query',weight=0.0)
        
        wanted_ents = set(mentioned_ents)
        wanted_rels = set(mentioned_rels)

        early_exit = False
        if (len(wanted_ents)>3 and len(wanted_rels)>0 and self.dataset=='musique'):
            early_exit = True
        early_exit = False

        wccs = list(nx.connected_components(g))
        sorted_wccs = sorted(wccs,key=len,reverse=True)
        cand_ctxs_lists = list()

        for i in range(len(sorted_wccs)):
            if (early_exit) and (len(wanted_ents)==0):
                break
            cand_ctxs_list = []
            wcc = sorted_wccs[i]
            for ent in wcc:
                if ent in wanted_ents:
                    wanted_ents.remove(ent)
            if len(wcc)>1:
                subgraph = g.subgraph(wcc)
                mst = nx.maximum_spanning_tree(subgraph,weight='weight')
                cand_ctx_list = []
                for cand_edge in mst.edges(data=True):
                    if cand_edge[2]['source']=='query':
                        continue
                    else:
                        cand_ctx_list.append(cand_edge[2]['source'])
                cand_ctxs_list.append(cand_ctx_list)
            else:
                sorted_edges = sorted(g.subgraph(wcc).edges(data=True),key=lambda x:x[2]['weight'],reverse=True)
                for edge in sorted_edges:
                    if edge[2]['source']=='query':
                        continue
                    else:
                        cand_ctxs_list.append([edge[2]['source'],])
                        break
            cand_ctxs_lists.append(cand_ctxs_list)

        cand_ids_lists = list()
        for cand_ctxs_list in cand_ctxs_lists:
            cand_ids_lists.extend(cand_ctxs_list)
        
        cand_tpts = []
        cand_strs = []
        for cand_ids_list in cand_ids_lists:
            ctx_str = ''
            tpt_str = ''
            for cand_id in cand_ids_list:
                if self.dataset=='hotpotqa':
                    cand_ent,seq_str = cand_id.split('##')
                    idx_seq_str = seq_str
                elif self.dataset=='musique':
                    idx_str,cand_ent,seq_str = cand_id.split('##')
                    idx_seq_str = f'{idx_str}##{seq_str}'
                ctx_str += self.chunks_index[cand_ent][idx_seq_str]
                if (self.use_tpt) and (cand_ent in self.doc2kg) and (idx_seq_str in self.doc2kg[cand_ent]) and (len(self.doc2kg[cand_ent][idx_seq_str])>0):
                    tpt_str += ', '.join([f'{h} has/is {r} {t}' for h,r,t in self.doc2kg[cand_ent][idx_seq_str][:min(len(self.doc2kg[cand_ent][idx_seq_str]),3)]])
                    if len(tpt_str)>0:
                        ctx_str = f'{ctx_str} Relational facts: {tpt_str}.'
            cand_strs.append(ctx_str)
            cand_tpts.append(tpt_str)
        

        if self.native_scores is not None:
            return self._select_native(nodes, query_bundle, cand_ids_lists, cand_strs)

        if len(cand_strs)==0:
            scores = self.reranker.compute_score([(query_bundle.query_str,node.node.text) for node in nodes])
            sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
            wanted_nodes = [nodes[sorted_seqs[i]] for i in range(min(self.topk,len(sorted_seqs)))]
            return wanted_nodes
        
        wanted_ctxs = []
        scores = self.reranker.compute_score([(query_bundle.query_str,cand_str) for cand_str in cand_strs])
        # scores = self.reranker.compute_score([(query_bundle.query_str,cand_tpt) for cand_tpt in cand_tpts])
        sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
        for seq in sorted_seqs:
            if len(set(wanted_ctxs)|set(cand_ids_lists[seq]))>self.topk:
                break
            wanted_ctxs.extend(cand_ids_lists[seq])
            wanted_ctxs = list(set(wanted_ctxs))

        if len(wanted_ctxs)<self.topk//2:
            cands = [(query_bundle.query_str,node.node.text,) for node in nodes if node.node.id_ not in wanted_ctxs]
            if len(cands)>0:
                scores = self.reranker.compute_score(cands)
                sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
                for seq in sorted_seqs[:self.topk]:
                    if nodes[seq].node.id_ not in wanted_ctxs:
                        wanted_ctxs.append(nodes[seq].node.id_)
                        if len(wanted_ctxs)>=self.topk:
                            break

        wanted_nodes = []
        for node in nodes:
            if node.node.id_ in wanted_ctxs:
                wanted_nodes.append(node)
        return wanted_nodes

class SemanticFilterPostProcessor(BaseNodePostprocessor):

    "KnowledgeGraph-based Node Processor"
    encoder: SentenceTransformer
    threshold: float

    @classmethod
    def class_name(cls):
        return "SemanticFilterPostProcessor"

    def _postprocess_nodes(self, nodes, query_bundle=None):
        if not nodes:
            return []

        question = query_bundle.query_str
        chunks = [node.node.text for node in nodes]

        # Create question and chunk (matrix) vectors
        question_vector = self.encoder.encode(
            question,
            normalize_embeddings=True,
        )
        chunk_vectors = self.encoder.encode(
            chunks,
            normalize_embeddings=True,
        )

        # Matrix multiplication (one dot product per row / one similarity score per chunk)
        similarites = chunk_vectors @ question_vector

        kept_nodes = [
            node
            for node, score in zip(nodes, similarites)
            if score >= self.threshold
        ]

        # None pass threshold, keep the single sentence with the highest sematic similarity score
        if not kept_nodes:
            kept_nodes = [nodes[argmax(similarites)]]

        return kept_nodes

class CrossEncoderFilterPostProcessor(BaseNodePostprocessor):

    """Filter expanded sentences by joint question–sentence relevance scores."""
    encoder: CrossEncoder
    threshold: float

    @classmethod
    def class_name(cls):
       return "CrossEncoderFilterPostProcessor"


    def _postprocess_nodes(self, nodes, query_bundle=None):
        if not nodes:
            return []

        question = query_bundle.query_str
        pairs = [(question, node.node.text) for node in nodes]
        scores = self.encoder.predict(pairs)

        kept_nodes = [
            node
            for node, score in zip(nodes, scores)
            if score >= self.threshold
        ]

        # None pass threshold, keep the single sentence with the highest sematic similarity score
        if not kept_nodes:
            kept_nodes = [nodes[argmax(scores)]]

        return kept_nodes


def _rank_percentiles(scores):
    """Higher is better; give equal scores equal average ranks."""
    n = len(scores)
    if n <= 1:
        return np.ones(n)
    order = np.argsort(-scores, kind='stable')
    ranks = np.empty(n, dtype=float)
    start = 0
    while start < n:
        end = start + 1
        while end < n and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 1 - ((start + end - 1) / 2) / (n - 1)
        start = end
    return ranks


def select_joint_chunks(nodes, sem_scores, cross_scores, keep_k=10, cross_weight=0.7):
    """Select once using both rankings, always protecting the best CROSS match.

    Raw cosine values and logits are never averaged. Each model's scores become
    within-query rank percentiles; their weighted sum ranks the candidates.
    The top CROSS candidate is reserved, then remaining slots follow fused rank.
    Return original node objects in original order without replacing their scores.
    """
    if not isinstance(keep_k, int) or isinstance(keep_k, bool) or keep_k < 1:
        raise ValueError('joint keep_k must be a positive integer')
    if not np.isfinite(cross_weight) or not 0 <= cross_weight <= 1:
        raise ValueError('cross_weight must be finite and between 0 and 1')
    sem_scores = np.asarray(sem_scores, dtype=float).reshape(-1)
    cross_scores = np.asarray(cross_scores, dtype=float).reshape(-1)
    if len(sem_scores) != len(nodes) or len(cross_scores) != len(nodes):
        raise ValueError('Each candidate needs one SEM score and one CROSS score')
    if not np.all(np.isfinite(sem_scores)) or not np.all(np.isfinite(cross_scores)):
        raise ValueError('SEM and CROSS scores must be finite')
    if not nodes:
        return []
    fused = (cross_weight * _rank_percentiles(cross_scores) +
             (1 - cross_weight) * _rank_percentiles(sem_scores))
    # Tie breaks use each model separately, never an average of raw score scales.
    ranked = sorted(range(len(nodes)), key=lambda i: (-fused[i], -cross_scores[i], -sem_scores[i], i))
    anchor = max(range(len(nodes)), key=lambda i: (cross_scores[i], sem_scores[i], -i))
    selected = {anchor}
    for index in ranked:
        if len(selected) >= min(keep_k, len(nodes)):
            break
        selected.add(index)
    return [node for i, node in enumerate(nodes) if i in selected]


class JointFilterPostProcessor(BaseNodePostprocessor):
    """Score every expanded sentence with both encoders, then select once."""

    sem_encoder: SentenceTransformer
    cross_encoder: CrossEncoder
    keep_k: int = Field(default=10, ge=1)
    cross_weight: float = Field(default=0.7, ge=0, le=1)

    @classmethod
    def class_name(cls):
        return 'JointFilterPostProcessor'

    def _postprocess_nodes(self, nodes, query_bundle=None):
        if not nodes:
            return []
        if query_bundle is None:
            raise ValueError('Joint filtering requires a query')
        question = query_bundle.query_str
        texts = [node.node.text for node in nodes]
        query_vector = self.sem_encoder.encode(question, normalize_embeddings=True)
        chunk_vectors = self.sem_encoder.encode(texts, normalize_embeddings=True)
        sem_scores = chunk_vectors @ query_vector
        cross_scores = self.cross_encoder.predict([(question, text) for text in texts])
        kept = select_joint_chunks(nodes, sem_scores, cross_scores, self.keep_k, self.cross_weight)
        print(f'Joint selection: {len(nodes)} candidates -> {len(kept)} kept; CROSS weight={self.cross_weight}')
        return kept
