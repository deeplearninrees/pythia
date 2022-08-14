import random

def main():
    urls = open('../links.txt').read().splitlines()
    print(urls)
    random.shuffle(urls)
    with open('../links.txt', mode='w') as f:
        f.write('\n'.join(urls))

main()
