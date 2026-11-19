import os
import argparse
import tempfile
from pathlib import Path
import ujson as json
from tqdm import tqdm
from llama_index.llms.ollama import Ollama

def extract_triplets(llm, ctx):
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-"*20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{"-"*20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{"-"*20}\nText: {ctx}\nTriplets:'
    resp = llm.complete(query)
    resp = resp.text
    triplets = set()
    triplet_texts = resp.split('$$')
    for triplet_text in triplet_texts:
        if len(triplet_text) <= 6:
            continue
        triplet_text = triplet_text[1:-1]
        tokens = triplet_text.split('##')
        if not len(tokens) == 3:
            continue
        h = tokens[0].strip()
        r = tokens[1].strip()
        t = tokens[2].strip()
        if ('no ' in h) or ('no ' in t) or ('unknown' in h) or ('unknown' in t) or ('No ' in h) or ('No ' in t) or ('Unknown' in h) or ('Unknown' in t) or ('null' in h) or ('null' in t) or ('Null' in h) or ('Null' in t) or ('NULL' in h) or ('NULL' in t) or ('NO' in h) or ('NO' in r) or ('NO' in t) or (h==t):
            continue
        if (r not in ctx) and (t not in ctx):
            continue

        triplets.add((h, r, t))
    triplets = [[h,r,t] for (h,r,t) in triplets]
    return triplets

def main(args):
    """Extract each paragraph once; reuse complete files after interruption."""
    with open(args.data_path) as stream:
        data = json.load(stream)
    out_dir = Path(args.kg_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    filenames = {}
    for sample in data:
        for title, _ in sample['context']:
            filename = title.replace('/', '_') + '.json'
            if filename in filenames and filenames[filename] != title:
                raise ValueError(f'Graph filename collision: {title} and {filenames[filename]}')
            filenames[filename] = title
    llm = Ollama(model=args.model_name, request_timeout=args.timeout)
    count = 0
    for sample in tqdm(data, desc='Questions'):
        for ent, sentences in sample['context']:
            if ent in seen:
                continue
            seen.add(ent)
            out_path = out_dir / (ent.replace('/', '_') + '.json')
            if out_path.exists():
                # Do not silently skip a truncated/corrupt JSON file.
                with out_path.open() as stream:
                    json.load(stream)
                continue
            graph = {}
            for i, sentence in enumerate(tqdm(sentences, desc=ent[:40], leave=False)):
                ctx_text = sentence if i == 0 else f'{ent}: {sentence}'
                triples = extract_triplets(llm, ctx_text)
                if triples:
                    graph[i] = triples
            # The old extractor could leave a partial JSON on interruption.
            name = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', dir=out_dir, delete=False) as stream:
                    name = stream.name
                    json.dump(graph, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, out_path)
            finally:
                if name and os.path.exists(name):
                    os.unlink(name)
            count += 1
    print(f'Newly extracted paragraph KGs: {count}')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Extract HotpotQA paragraph graphs with Ollama')
    parser.add_argument('--data_path', default=str(root / 'data/hotpotqa/hotpot_dev_distractor_v1.json'))
    parser.add_argument('--kg_dir', default=str(root / 'data/hotpotqa/kgs/extract_subkgs'))
    parser.add_argument('--model_name', default='llama3:8b')
    parser.add_argument('--timeout', type=float, default=120)
    main(parser.parse_args())
