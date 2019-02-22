import pytest
import requests
import requests_mock
import os
import hashlib
import json
from checksumdir import dirhash

from contk._utils.file_utils import *

@pytest.fixture
def r_mock():
	with requests_mock.Mocker() as m:
		yield m

class TestFileUtils():
	def test_get_resource(self, r_mock):
		r_mock.get('http://coai.cs.tsinghua.edu.cn/', text='coai')

		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		res_path = get_resource_file_path('resources://coai', 'Default', cache_dir=cache_dir, config_dir=config_dir)
		
		assert(res_path == os.path.join(cache_dir, '6bd9bfb20a5159d1848a203ece33886690b15d785b0c5d632eed63d70442c58b'))
		assert(os.path.exists(res_path))

		hash_sha256 = hashlib.sha256()
		with open(res_path, "rb") as fin:
			for chunk in iter(lambda: fin.read(4096), b""):
				hash_sha256.update(chunk)
		assert(hash_sha256.hexdigest() == "146ce545f2ed0a8767aadae8f2921f7951df817b39b8f7d0db48bce87e3eaf69")

		meta_path = res_path + '.json'
		assert(os.path.exists(meta_path))
		with open(meta_path, 'r') as meta_file:
			meta = json.load(meta_file)
			assert(meta == {'local_path': res_path})

	def test_download_resource(self):
		cache_dir = './tests/_utils/dataset_cache'
		config_dir = './tests/_utils/dummy_coai'
		res_path = get_resource_file_path('http://coai.cs.tsinghua.edu.cn/file/mscoco.zip', 'MSCOCO', cache_dir=cache_dir, config_dir=config_dir)

		assert(res_path == os.path.join(cache_dir, 'eeb2fb42c3d6f5fef8caa8896c0302c3676965ae73b6cb7c11f831a8ae7c30d6_unzip/mscoco'))
		assert(os.path.exists(res_path))

		assert(dirhash(res_path, 'sha256') == 'f8ece190272864935f1849d784cb67d36b970c54aceadbcd7e845bdeefc23544')
		
		meta_path = os.path.join(cache_dir, 'eeb2fb42c3d6f5fef8caa8896c0302c3676965ae73b6cb7c11f831a8ae7c30d6.json')
		assert(os.path.exists(meta_path))
		with open(meta_path, 'r') as meta_file:
			meta = json.load(meta_file)
			assert(meta == {'local_path': res_path})