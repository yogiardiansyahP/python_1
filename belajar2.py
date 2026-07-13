import numpy as np

data = np.array([1,2,3,4,5])
print ("rata rata : ",np.mean(data))

mahasiswa = ["andi","budi","siti"]
nilai = {"andi":90,"budi": 80 ,"siti":70}

for nama in mahasiswa:
    print ("Nama:",nama)

    if nilai[nama]>=80:
        print("lulus")
    else:
        print("tidak lulus")