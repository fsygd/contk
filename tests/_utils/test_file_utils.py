import pytest
import requests
import os
import hashlib
import json

from contk._utils.file_utils import *

class TestFileUtils():
	def test_download(self, requests_mock):
		requests_mock.get('http://coai.cs.tsinghua.edu.cn/', text='coai')

		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		res_path = get_resource('coai', cache_dir=cache_dir, config_dir=config_dir)
		
		assert(res_path == os.path.join(cache_dir, 'coai'))
		assert(os.path.exists(res_path))

		hash_sha256 = hashlib.sha256()
		with open(res_path, "rb") as fin:
			for chunk in iter(lambda: fin.read(4096), b""):
				hash_sha256.update(chunk)
		assert(hash_sha256.hexdigest() == "aff6cd952797059eccb08bc61e91c86441d55920ddef5bdf98d164356ae2cff2")

		meta_path = res_path + '.json'
		assert(os.path.exists(meta_path))
		obj = json.load(meta_path)
		assert(obj == {'local_path': res_path})