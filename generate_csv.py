import names
from pprint import pprint
from uuid import uuid1
import random

COUNT = 7
import csv

C = 0



def write_to_csv(data):
    global C
    with open('allmemes_users_ratings.csv', mode='a') as f:
        fieldnames = ['user_id', 'meme_id', 'picture',  'rate', 'tags']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if C==0:
            writer.writeheader()
        user_id = data[0]
        meme_id = data[1]
        picture = data[2]
        rate = data[3]
        tags = tuple(data[4: len(data)])
        print(tags)
        tags = ', '.join(tags)
        #user_id, meme_id, picture, rate, tags = data
        writer.writerow({'user_id': user_id,
                         'meme_id': meme_id,
                         'picture': picture,
                         'rate': rate,
                         'tags': tags})
        #f = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        #f.writerow(data)
        C += 1

def get_list_of_id():
    return [i for i in range(COUNT)]


def get_name_of_picture():
    return f'public/memes/{names.get_first_name()}'


def get_list_of_picture():
    return [get_name_of_picture() for _ in range(COUNT)]


def get_hundrend_tags():
    return [str(uuid1().hex) for _ in range(100)]


def generate_rate():
    return random.randint(0, 3)


def get_list_of_rates():
    return [generate_rate() for _ in range(COUNT)]

def main():
    list_of_id = get_list_of_id()
    list_of_meme_id = get_list_of_id()
    list_of_picture = get_list_of_picture()
    list_of_hundred_tags = get_hundrend_tags()
    list_of_rates = get_list_of_rates()
    for i in zip(list_of_id, list_of_meme_id, list_of_picture, list_of_rates):
        i = list(i)
        for number in range(random.randint(0, 10)):
            tmp = list_of_hundred_tags[random.randint(0, 10)]
            i.append(tmp)
        i = tuple(i)
        write_to_csv(i)





if __name__ == '__main__':
    main()