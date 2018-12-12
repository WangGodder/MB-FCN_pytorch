import os


def change_file(file_url, data_dir, out_url=""):
    f = open(file_url, 'r')
    if out_url == "":
        out_url = "wider.list"
    w = open(out_url, 'w')
    file_name = f.readline().strip("\n")
    while file_name != '':
        out = os.path.join(data_dir, file_name)  # type: str
        face_num = f.readline().strip("\n")
        out += " " + face_num
        for _ in range(0, int(face_num)):
            localtion = f.readline() # type: str
            for c in localtion.split(" ")[:4]:
                out += " " + c
        w.write(out + "\n")
        print(out + "\n")
        file_name = f.readline().strip("\n")
    f.close()
    w.close()


if __name__ == '__main__':
    file_url = "F:\datasets\widerface\wider_face_split\wider_face_train_bbx_gt.txt"
    data_dir = "F:\datasets\widerface\WIDER_train\images"
    out_url = "F:\datasets\widerface\\train_winder.list"
    change_file(file_url, data_dir, out_url)