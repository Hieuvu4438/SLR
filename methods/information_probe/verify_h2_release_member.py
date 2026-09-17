"""Verify existing H2S bytes against the pinned public CiCo release; no inference."""
import hashlib
import json
import os
import time
import traceback
import urllib.request
import zipfile

from .common import ART, ROOT, dump, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'
URL = 'https://drive.usercontent.google.com/download?id=1Hpcn5obCcG5JHa3nLvHqX9pfrp7g6wDu&export=download&confirm=t'
ARCHIVE_SHA = 'f02ea0b2a64123b2c8386ccb467c00143454ee6405ed4c07e074dcc712224c6c'
ARCHIVE_BYTES = 971005695
MEMBERS = {'H2S_sota.pth': '67f557976d523dc3cd63cae73c29c88145e409b77221c90260ae51b9d0c95b35',
           'ph_sota.pth': '598b822aafd1b3cd767b290410ea1191f723dc87dd096775c3670e224b6e7b9f'}


def main():
    path = OUT/'H2S-RELEASE-MEMBER-VERIFY_run.json'
    dest = ART/'H2S-release-audit'
    if path.exists() or dest.exists():
        raise FileExistsError('Preserve earlier provenance attempts')
    dest.mkdir()
    started = time.monotonic()
    result = {'status': 'running', 'pid': os.getpid(), 'code_sha256': sha(__file__),
              'public_url': URL, 'expected_archive_sha256': ARCHIVE_SHA,
              'expected_archive_bytes': ARCHIVE_BYTES, 'bytes_downloaded': 0,
              'new_training_or_inference': False, 'test_data_read': False,
              'training_lineage_verified': False, 'members': {}}

    def record():
        result['wall_seconds'] = time.monotonic()-started
        dump(path, result)

    def timeout_check():
        if time.monotonic()-started > 300:
            raise TimeoutError('300-second provenance audit timeout')

    record()
    try:
        provenance = ROOT/'artifacts/pretrained/cico_checkpoint_provenance.json'
        old = json.loads(provenance.read_text())
        assert old['archive']['sha256'] == ARCHIVE_SHA and old['archive']['bytes'] == ARCHIVE_BYTES
        result['old_provenance_sha256'] = sha(provenance)
        assert all(sha(ROOT/'artifacts/pretrained'/name) == h for name, h in MEMBERS.items())
        archive = dest/'final_models.zip'
        result['archive_path'] = str(archive)
        digest = hashlib.sha256()
        last_record = started
        with urllib.request.urlopen(URL, timeout=30) as response, archive.open('xb') as output:
            result['http_status'] = response.status
            result['content_type'] = response.headers.get('Content-Type')
            result['content_length'] = int(response.headers['Content-Length'])
            assert response.status == 200 and result['content_length'] == ARCHIVE_BYTES
            while block := response.read(8*1024*1024):
                timeout_check()
                result['bytes_downloaded'] += len(block)
                assert result['bytes_downloaded'] <= ARCHIVE_BYTES
                output.write(block)
                digest.update(block)
                if time.monotonic()-last_record >= 5:
                    record()
                    last_record = time.monotonic()
        result['archive_sha256'] = digest.hexdigest()
        assert result['bytes_downloaded'] == ARCHIVE_BYTES and digest.hexdigest() == ARCHIVE_SHA
        # Read exactly the registered members as streams; never extract paths or execute pickle.
        with zipfile.ZipFile(archive) as zipped:
            result['archive_member_names'] = zipped.namelist()
            for name, expected in MEMBERS.items():
                matches = [i for i in zipped.infolist() if i.filename == name or i.filename.endswith('/'+name)]
                assert len(matches) == 1 and matches[0].file_size == 350469151
                digest, count = hashlib.sha256(), 0
                with zipped.open(matches[0]) as member:
                    while block := member.read(8*1024*1024):
                        timeout_check()
                        count += len(block)
                        assert count <= 350469151
                        digest.update(block)
                assert count == 350469151 and digest.hexdigest() == expected
                result['members'][name] = {'archive_member': matches[0].filename, 'bytes': count,
                    'sha256': digest.hexdigest(), 'local_file_sha256': sha(ROOT/'artifacts/pretrained'/name),
                    'archive_and_local_exact': True}
                record()
        result.update(status='completed', byte_provenance_verified=True)
        record()
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc())
        record()
        raise


if __name__ == '__main__':
    main()
