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
import hashlib

from tqdm import tqdm
from checksumdir import dirhash

import requests

class Downloader():
	r'''Base class of Downloader
	'''
	def __init__(self):
		logging.basicConfig(level=logging.INFO, \
				format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		self.logger = logging.getLogger(__name__)

		self.dataset_cache_path = os.path.join(os.getenv('HOME'), 'dataset_cache')
		self.config_dir = '.'
		os.makedirs(self.dataset_cache_path, exist_ok=True)

	def get_config(self, res_name):
		r'''Get config(dict) by the name of resource
		'''
		config_path = os.path.join(self.config_dir, res_name + '.json')
		if not os.path.exists(config_path):
			raise FileNotFoundError("file {} not found".format(config_path))
		with open(config_path, 'r') as config_file:
			config = json.load(config_file)

		return config


	def http_get(self, url, temp_file):
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


	def preprocess(self, local_path):
		"""
		preprocess after downloading and before saving
		"""
		return local_path


	def postprocess(self, local_path):
		"""
		process before read
		"""
		return local_path


	def get_file_sha256(self, file_path):
		""" get sha256 of given file """
		hash_sha256 = hashlib.sha256()
		with open(file_path, "rb") as fin:
			for chunk in iter(lambda: fin.read(4096), b""):
				hash_sha256.update(chunk)
		return hash_sha256.hexdigest()


	def get_hashtag(self, file_path):
		""" get sha256 of given directory or file """
		if os.path.isdir(file_path):
			return dirhash(file_path, 'sha256')
		else:
			return self.get_file_sha256(file_path)


	def get_resource(self, res_name):
		"""
		Get the resource with the given name.
		If not cached, download it using the URL stored in config file.
		If cached, check the hashtag.
		"""
		config = self.get_config(res_name)

		cache_dir = self.dataset_cache_path
		cache_path = os.path.join(cache_dir, res_name)
		meta_path = cache_path + '.json'

		if not os.path.exists(meta_path):
			with tempfile.NamedTemporaryFile() as temp_file:
				url = config['link']
				self.logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

				self.http_get(url, temp_file)

				# flush to avoid truncation
				temp_file.flush()
				# shutil.copyfileobj() starts at the current position
				temp_file.seek(0)

				self.logger.info("copying %s to cache at %s", temp_file.name, cache_path)
				with open(cache_path, 'wb') as cache_file:
					shutil.copyfileobj(temp_file, cache_file)
				self.logger.info("removing temp file %s", temp_file.name)

				self.logger.info("preprocessing ...")
				cache_path = self.preprocess(cache_path)

				cache_hashtag = self.get_hashtag(cache_path)

				if cache_hashtag == config['hashtag']:
					self.logger.info("hashtag %s checked", cache_hashtag)

					self.logger.info("creating metadata file for %s", cache_path)
					meta = {'local_path': cache_path}
					with open(meta_path, 'w') as meta_file:
						json.dump(meta, meta_file)
				else:
					self.logger.info("local hashtag %s differs with standard %s", cache_hashtag, config['hashtag'])
					raise ValueError("bad hashtag of {}".format(res_name))
		else:
			with open(meta_path, 'r') as meta_file:
				meta = json.load(meta_file)
				cache_path = meta['local_path']

			cache_hashtag = self.get_hashtag(cache_path)

			if cache_hashtag == config['hashtag']:
				self.logger.info("hashtag %s checked", cache_hashtag)
			else:
				self.logger.info("local hashtag %s differs with %s", cache_hashtag, config['hashtag'])
				raise ValueError("bad hashtag of {}".format(res_name))

		return self.postprocess(cache_path)


	def import_local_benchmark(self, res_name, local_path):
		"""
		import benchmark from local, if hashtag checked, save to cache.
		"""
		config = self.get_config(res_name)

		local_hashtag = self.get_hashtag(local_path)
		if local_hashtag == config['hashtag']:
			self.logger.info("hashtag %s checked", local_hashtag)

			self.logger.info("creating metadata file for %s", local_path)
			meta = {'local_path': local_path}
			meta_path = os.path.join(self.dataset_cache_path, res_name) + '.json'
			with open(meta_path, 'w') as meta_file:
				json.dump(meta, meta_file)

			return self.postprocess(local_path)
		else:
			self.logger.info("local hashtag %s differs with standard %s", local_hashtag, config['hashtag'])
			raise ValueError("bad hashtag of {}".format(res_name))


	def import_local_resource(self, local_path):
		"""
		import temporary resources from local
		"""
		return self.postprocess(local_path)
