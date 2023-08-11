import string
import zipfile
from copy import copy

from tqdm import tqdm
from itertools import permutations

zip_file = r"C:\Users\whats\Downloads\111.zip"
# zip_file = r"C:\Users\whats\Downloads\messages.zip"

max_suffix = 3
pass_dict = "!@#$%^&*()"
# pass_dict = "0123456789)!@#$%^&*("

def generate_password():
    results = []
    suffixes = ["".join(item) for item in permutations(pass_dict, max_suffix) if not all([item1.isdigit() for item1 in item])]
    for item1 in ["qq", "QQ"]:
        str = ""
        str += item1
        str += ""
        for item2 in ["lyl", "LYL", "LyL"]:
            str2 = copy(str)
            str2 += item2
            for item3 in suffixes:
                str3 = copy(str2)
                str3 += item3
                results.append(str3)
    return results


if __name__ == '__main__':
    passwords = generate_password()
    print("Total passwords to test:", len(passwords))
    zip_file = zipfile.ZipFile(zip_file)

    for word in tqdm(passwords):
        try:
            zip_file.extractall(pwd=word.encode())
        except Exception as e:
            continue
        else:
            print("[+] Password found:", word)
            exit(0)
    print("[!] Password not found, try other wordlist.")
