
mahasiswa = ["andi","budi","siti","dina","eko","fajar","gita","hadi","indah","joko"]
nilai = {"andi":85,"budi":74,"dina":50,"siti":92,"eko":45,"fajar":77,"gita":80,"hadi":59,"indah":90,"joko":6}


total_nilai = sum(nilai.values())
jumlah_mahasiswa = len(nilai)
rata_rata = total_nilai / jumlah_mahasiswa

print("Rata-rata nilai kelas =", rata_rata)

nilai_tertinggi = max(nilai, key=nilai.get)
nilai_terendah = min(nilai, key=nilai.get)

print("Nilai tertinggi :", nilai_tertinggi, "-", nilai[nilai_tertinggi])
print("Nilai terendah  :", nilai_terendah, "-", nilai[nilai_terendah])


jumlah_di_atas_rata = 0

for nama, skor in nilai.items():
    if skor > rata_rata:
        jumlah_di_atas_rata += 1

print("Jumlah mahasiswa di atas rata-rata =", jumlah_di_atas_rata)

lulus = 0
tidak_lulus = 0

for nama in mahasiswa:
    print("Nama:", nama)

    if nilai[nama] >= 80:
        print("A - Lulus")
        lulus += 1
    elif nilai[nama] >= 70:
        print("B - Lulus")
        lulus += 1
    elif nilai[nama] >= 60:
        print("C - Lulus")
        lulus += 1
    elif nilai[nama] >= 50:
        print("D - Tidak Lulus")
        tidak_lulus += 1
    else:
        print("E - Tidak Lulus")
        tidak_lulus += 1

print("\nJumlah mahasiswa lulus =", lulus)
print("Jumlah mahasiswa tidak lulus =", tidak_lulus)