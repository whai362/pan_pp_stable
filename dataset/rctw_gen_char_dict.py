import mmcv

rctw_17_root = '../data/RCTW-17/'
rctw_17_train_root = rctw_17_root + 'train/'

train_list = mmcv.list_from_file(rctw_17_root + 'train_list.txt')
dictmap_to_lower = mmcv.load('dictmap_to_lower.json')

charset = set()
for i, image_name in enumerate(train_list):
    print(i)
    gt_name = image_name.replace('.jpg', '.txt')

    lines = mmcv.list_from_file(rctw_17_train_root + gt_name)

    for line in lines:
        text = line.split('\"')[-2]
        for c in text:
            if c in dictmap_to_lower:
                c = dictmap_to_lower[c]
            charset.add(c)
char_dict = {
    'char2id': {},
    'id2char': {}
}
charset = list(charset)
charset.sort()
for i, c in enumerate(charset):
    char_dict['char2id'][c] = i
    char_dict['id2char'][i] = c

print(char_dict)
print(len(charset))

mmcv.dump(char_dict, rctw_17_root + 'char_dict.json')
