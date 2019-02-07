"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import os
import logging
import json
import tempfile
import shutil
from typing import IO
import hashlib

from tqdm import tqdm
from checksumdir import dirhash

import requests

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_CACHE_PATH = os.path.join(os.getenv('HOME'), 'dataset_cache')
CONFIG_DIR = '.'
os.makedirs(DATASET_CACHE_PATH, exist_ok=True)

def get_config(res_name: str):
    """
    Get config(dict) by the name of resource
    """
    config_path = os.path.join(CONFIG_DIR, res_name + '.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError("file {} not found".format(config_path))
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    return config


def http_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from http"""
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def preprocess(local_path: str, res_type) -> str:
    """
    preprocess after downloading and before saving
    """
    # todo
    return local_path


def postprocess(local_path: str, res_type) -> str:
    """
    process before read
    """
    # todo
    return local_path


def get_file_md5(file_path: str) -> str:
    """ get md5 of given file """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_hashtag(file_path: str) -> str:
    """ get md5 of given directory or file """
    if os.path.isdir(file_path):
        return dirhash(file_path, 'md5')
    else:
        return get_file_md5(file_path)


def get_resource(res_name: str):
    """
    Get the resource with the given name.
    If not cached, download it using the URL stored in config file.
    If cached, check the hashtag.
    """
    config = get_config(res_name)
    
    cache_dir = DATASET_CACHE_PATH
    cache_path = os.path.join(cache_dir, res_name)
    meta_path = cache_path + '.json'
    res_type = config['type']

    if not os.path.exists(meta_path):
        with tempfile.NamedTemporaryFile() as temp_file:
            url = config['link']
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)
            
            http_get(url, temp_file)

            # flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
            logger.info("removing temp file %s", temp_file.name)
            
            logger.info("preprocessing ...")
            cache_path = preprocess(cache_path, res_type)

            cache_hashtag = get_hashtag(cache_path)
            
            if cache_hashtag == config['hashtag']:
                logger.info("hashtag %s checked", cache_hashtag)
   
                logger.info("creating metadata file for %s", cache_path)
                meta = {'local_path': cache_path}
                with open(meta_path, 'w') as meta_file:
                    json.dump(meta, meta_file)
            else:
                logger.info("local hashtag %s differs with standard %s", cache_hashtag, config['hashtag'])
                raise ValueError("bad hashtag of {}".format(res_name))
    else:
        with open(meta_path, 'r') as meta_file:
            meta = json.load(meta_file)
            cache_path = meta['local_path']
        
        cache_hashtag = get_hashtag(cache_path)

        if cache_hashtag == config['hashtag']:
            logger.info("hashtag %s checked", cache_hashtag)
        else:
            logger.info("local hashtag %s differs with standard %s", cache_hashtag, config['hashtag'])
            raise ValueError("bad hashtag of {}".format(res_name))
    
    return postprocess(cache_path, config['type'])


def import_local_benchmark(res_name: str, local_path: str) -> str:
    """
    import benchmark from local, if hashtag checked, save to cache.
    """
    config = get_config(res_name)

    local_hashtag = get_hashtag(local_path)
    if local_hashtag == config['hashtag']:
        logger.info("hashtag %s checked", local_hashtag)

        logger.info("creating metadata file for %s", local_path)
        meta = {'local_path': local_path}
        meta_path = os.path.join(DATASET_CACHE_PATH, res_name) + '.json'
        with open(meta_path, 'w') as meta_file:
            json.dump(meta, meta_file)

        return postprocess(local_path, config['type'])
    else:
        logger.info("local hashtag %s differs with standard %s", local_hashtag, config['hashtag'])
        raise ValueError("bad hashtag of {}".format(res_name))


def import_local_resource(local_path: str, res_type) -> str:
    """
    import temporary resources from local
    """
    return postprocess(local_path, res_type)
