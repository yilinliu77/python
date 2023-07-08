import os
import time
import urllib3
from urllib.request import urlopen

import py7zr
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# pip install py7zr bs4 requests tqdm
urllib3.disable_warnings()

id_start = 0

s = requests.Session()
s.headers.update({
    "User-Agent": "'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'",
})

abc_links = [
    "https://archive.nyu.edu/handle/2451/44309",
    "https://archive.nyu.edu/handle/2451/44310",
    "https://archive.nyu.edu/handle/2451/44318",
    "https://archive.nyu.edu/handle/2451/44319",
    "https://archive.nyu.edu/handle/2451/44320",
    "https://archive.nyu.edu/handle/2451/44321",
    "https://archive.nyu.edu/handle/2451/44322",
    "https://archive.nyu.edu/handle/2451/44323",
    "https://archive.nyu.edu/handle/2451/44324",
    "https://archive.nyu.edu/handle/2451/44325",
    "https://archive.nyu.edu/handle/2451/44326",
    "https://archive.nyu.edu/handle/2451/44327",
    "https://archive.nyu.edu/handle/2451/44328",
    "https://archive.nyu.edu/handle/2451/44329",
    "https://archive.nyu.edu/handle/2451/44330",
    "https://archive.nyu.edu/handle/2451/44331",
    "https://archive.nyu.edu/handle/2451/44332",
    "https://archive.nyu.edu/handle/2451/44333",
    "https://archive.nyu.edu/handle/2451/44334",
    "https://archive.nyu.edu/handle/2451/44335",
    "https://archive.nyu.edu/handle/2451/44337",
    "https://archive.nyu.edu/handle/2451/44338",
    "https://archive.nyu.edu/handle/2451/44339",
    "https://archive.nyu.edu/handle/2451/44340",
    "https://archive.nyu.edu/handle/2451/44341",
    "https://archive.nyu.edu/handle/2451/44342",
    "https://archive.nyu.edu/handle/2451/44343",
    "https://archive.nyu.edu/handle/2451/44344",
    "https://archive.nyu.edu/handle/2451/44345",
    "https://archive.nyu.edu/handle/2451/44346",
    "https://archive.nyu.edu/handle/2451/44347",
    "https://archive.nyu.edu/handle/2451/44348",
    "https://archive.nyu.edu/handle/2451/44349",
    "https://archive.nyu.edu/handle/2451/44350",
    "https://archive.nyu.edu/handle/2451/44351",
    "https://archive.nyu.edu/handle/2451/44352",
    "https://archive.nyu.edu/handle/2451/44353",
    "https://archive.nyu.edu/handle/2451/44354",
    "https://archive.nyu.edu/handle/2451/44355",
    "https://archive.nyu.edu/handle/2451/44356",
    "https://archive.nyu.edu/handle/2451/44357",
    "https://archive.nyu.edu/handle/2451/44358",
    "https://archive.nyu.edu/handle/2451/44359",
    "https://archive.nyu.edu/handle/2451/44360",
    "https://archive.nyu.edu/handle/2451/44361",
    "https://archive.nyu.edu/handle/2451/44362",
    "https://archive.nyu.edu/handle/2451/44363",
    "https://archive.nyu.edu/handle/2451/44365",
    "https://archive.nyu.edu/handle/2451/44366",
    "https://archive.nyu.edu/handle/2451/44367",
    "https://archive.nyu.edu/handle/2451/44368",
    "https://archive.nyu.edu/handle/2451/44369",
    "https://archive.nyu.edu/handle/2451/44370",
    "https://archive.nyu.edu/handle/2451/44371",
    "https://archive.nyu.edu/handle/2451/44372",
    "https://archive.nyu.edu/handle/2451/44373",
    "https://archive.nyu.edu/handle/2451/44374",
    "https://archive.nyu.edu/handle/2451/44375",
    "https://archive.nyu.edu/handle/2451/44376",
    "https://archive.nyu.edu/handle/2451/44378",
    "https://archive.nyu.edu/handle/2451/44379",
    "https://archive.nyu.edu/handle/2451/44380",
    "https://archive.nyu.edu/handle/2451/44381",
    "https://archive.nyu.edu/handle/2451/44382",
    "https://archive.nyu.edu/handle/2451/44383",
    "https://archive.nyu.edu/handle/2451/44384",
    "https://archive.nyu.edu/handle/2451/44385",
    "https://archive.nyu.edu/handle/2451/44386",
    "https://archive.nyu.edu/handle/2451/44387",
    "https://archive.nyu.edu/handle/2451/44388",
    "https://archive.nyu.edu/handle/2451/44389",
    "https://archive.nyu.edu/handle/2451/44390",
    "https://archive.nyu.edu/handle/2451/44391",
    "https://archive.nyu.edu/handle/2451/44392",
    "https://archive.nyu.edu/handle/2451/44393",
    "https://archive.nyu.edu/handle/2451/44394",
    "https://archive.nyu.edu/handle/2451/44395",
    "https://archive.nyu.edu/handle/2451/44396",
    "https://archive.nyu.edu/handle/2451/44406",
    "https://archive.nyu.edu/handle/2451/44407",
    "https://archive.nyu.edu/handle/2451/44408",
    "https://archive.nyu.edu/handle/2451/44409",
    "https://archive.nyu.edu/handle/2451/44410",
    "https://archive.nyu.edu/handle/2451/44411",
    "https://archive.nyu.edu/handle/2451/44412",
    "https://archive.nyu.edu/handle/2451/44413",
    "https://archive.nyu.edu/handle/2451/44414",
    "https://archive.nyu.edu/handle/2451/44415",
    "https://archive.nyu.edu/handle/2451/44416",
    "https://archive.nyu.edu/handle/2451/44417",
    "https://archive.nyu.edu/handle/2451/44397",
    "https://archive.nyu.edu/handle/2451/44398",
    "https://archive.nyu.edu/handle/2451/44399",
    "https://archive.nyu.edu/handle/2451/44400",
    "https://archive.nyu.edu/handle/2451/44401",
    "https://archive.nyu.edu/handle/2451/44402",
    "https://archive.nyu.edu/handle/2451/44403",
    "https://archive.nyu.edu/handle/2451/44404",
    "https://archive.nyu.edu/handle/2451/44405",
    "https://archive.nyu.edu/handle/2451/44418",

]


def download_from_url(url, dst):
    header_results = s.head(url)
    file_size = int(header_results.headers.get('Content-Length', -1))
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte == file_size:
        return file_size
    if os.path.exists(dst):
        os.remove(dst)
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=0,
        unit='B', unit_scale=True, desc=dst.split('\\')[-1])
    req = s.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


def download_and_extract(url, output_filename, new_folder):
    download_from_url(url, output_filename)

    print("Start to extract {}".format(output_filename))
    # Create new folder if it doesn't exist
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    with py7zr.SevenZipFile(output_filename, mode='r') as z:
        missing_file = []
        for item in z.list():
            if not os.path.exists(os.path.join(new_folder, item.filename)):
                missing_file.append(item.filename)
            elif os.path.getsize(os.path.join(new_folder, item.filename)) != item.uncompressed and not item.is_directory:
                missing_file.append(item.filename)
        z.extract(new_folder, missing_file)
    print("{} extract done".format(output_filename))


def process(v_item):
    a_tag = soup.find('a', string=v_item)
    if a_tag is not None:
        hyperlink = "https://archive.nyu.edu" + a_tag.get('href')
        download_and_extract(
            hyperlink, v_item, "data")
    else:
        print("Cannot find {} in {}".format(v_item, link))
        exit(1)


if __name__ == '__main__':
    for i_link, link in enumerate(abc_links[id_start:]):
        prefix = "{:04d}".format(i_link+id_start)
        print("Start to process {}".format(prefix))
        while True:
            try:
                results = s.get(link)
                break
            except Exception as e:
                print(e)
                print("Catched exception, wait 20s")
                time.sleep(20)

        soup = BeautifulSoup(results.content, 'html.parser')

        # Feat
        prefix1 = 'abc_{}_feat_v00.7z'.format(prefix)
        while True:
            try:
                process(prefix1)
                break
            except Exception as e:
                print(e)
                print("Catched exception, wait 20s")
                time.sleep(20)

        # Meta
        prefix1 = 'abc_{}_meta_v00.7z'.format(prefix)
        while True:
            try:
                process(prefix1)
                break
            except Exception as e:
                print(e)
                print("Catched exception, wait 20s")
                time.sleep(20)

        # Obj
        while True:
            prefix1 = 'abc_{}_obj_v00.7z'.format(prefix)
            try:
                process(prefix1)
                break
            except Exception as e:
                print(e)
                print("Catched exception, wait 20s")
                time.sleep(20)

        print("{} done".format(prefix))
