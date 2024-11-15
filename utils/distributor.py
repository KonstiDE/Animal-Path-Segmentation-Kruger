import math
import os
import glob
import random
import shutil

import config.config as cfg


def move_into_split():
    splits = cfg.get_split_map()
    versions = cfg.get_stolsnek_features().keys()

    list_of_lists = [glob.glob(os.path.join(cfg.get_frame_path(), str(version), "df_part*.npz")) for version in versions]
    dict_of_lists = dict(zip(versions, list_of_lists))

    dict_of_lists_cert = dict(zip(versions, [[] for i in range(len(versions))]))

    for version, partial_list in dict_of_lists.items():
        for file in partial_list:
            filename = os.path.basename(file)

            exists_everywhere = True
            for subversion in [x for x in versions if x != version]:
                if not os.path.isfile(os.path.join(cfg.get_frame_path(), subversion, filename)):
                    exists_everywhere = False
                    break

            if exists_everywhere:
                for v in versions:
                    filename_list = [os.path.basename(f) for f in dict_of_lists_cert[v]]

                    if filename not in filename_list:
                        dict_of_lists_cert[v].append(os.path.join(cfg.get_frame_path(), v, filename))


    tpv = len(dict_of_lists_cert[list(versions)[0]])
    permutation = list(range(tpv))
    random.shuffle(permutation)

    for version in dict_of_lists_cert.keys():
        dict_of_lists_cert[version] = [dict_of_lists_cert[version][i] for i in permutation]

    train_percent = math.floor(tpv * splits["train"][0])
    validation_percent = math.floor(tpv * splits["validation"][0])
    test_percent = tpv - train_percent - validation_percent


    for version, partial_list in dict_of_lists_cert.items():
        for i in range(train_percent):
            file = partial_list[i]
            shutil.move(
                file, os.path.join(cfg.get_frame_path(), str(version), "train", os.path.basename(str(file)))
            )
        for i in range(validation_percent):
            file = partial_list[i + train_percent]
            shutil.move(
                file, os.path.join(cfg.get_frame_path(), str(version), "validation", os.path.basename(str(file)))
            )
        for i in range(test_percent):
            file = partial_list[i + train_percent + validation_percent]
            shutil.move(
                file, os.path.join(cfg.get_frame_path(), str(version), "test", os.path.basename(str(file)))
            )



if __name__ == '__main__':
    move_into_split()

