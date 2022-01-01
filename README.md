# SZE img digital processing

## Épületek, objektumok felismerése műholdképeken

==========================================================================




I. Rendszerkövetelmények

- 32 ill. 64 bites rendszer ill. dedikált GPU ajánlott. 

- Kompatibilitás tesztelve Windows 10 x64 OS-en.
- Kompatibilitás feltételezhető Unix-rendszereken is, de nem garantált.

- Python 3.7.6 verziója ajánlott teszteléshez. 3.5-3.9 közötti verziókkal is vélhetően működik, de dependencia-problémák előfordulhatnak.

- Az alábbi python library-k kerültek feltelepítésre: 

`cycler          0.11.0
kiwisolver      1.3.2
matplotlib      3.5.0
numpy           1.21.4
opencv-python   4.5.4.60
packaging       21.3
Pillow          8.4.0
pip             21.1.1
pyparsing       3.0.6
python-dateutil 2.8.2
setuptools      49.2.1
setuptools-scm  6.3.2
six             1.16.0
tomli           1.2.2
pip             21.3.1
setuptools      41.2.0
tensorflow      2.5.0
wrapt           1.13.0`

A program lineáris lefutású Python-kód, Oo alkalmazása nélkül. OpenCV-t használ, így az általa elfogadott raszter-képformátumokkal dolgozhatunk. 0-255 intenzitás közötti értékeket használunk.


=====================================


II. Technikai dokumentáció


A képfeldolgozás lépései:

1. Kép kicsinyítése (képfeldolgozás erőforrás-hatékonyságának növelése céljából)

2. Zöldterületek kiszűrése, maszkolása (pl. szürkeárnyalatban cseréptetőhöz hasonló értékű erdők, bokros területek eliminálásához)

3. Szürkeárnyalatos képpé konvertálás

4. Gauss-féle elmosás

5. Bináris treshold alkalmazása - adott minimum ill. maximum képpontintenzitás-határok között szeretnénk feldolgozni a kép tartalmát. Az ezen kívül eső tartományok egységesen feketék lesznek.

6. Kontúrok azonosítása

7. Kontúrok közül a túlságosan kis méretű objektumok (pl. autó, kuka) ill. túl nagy méretű objektumok (pl. szintén sötét színű, szabályos alakú folyók) kiszűrése, kontúr kerülete ill. körülhatárolt területe alapján

8. Objektum külső kontúr-vonalainak egymással bezárt szöge alapján fals pozitívak kiszűrése (ideális esetben az épületek közel derékszögű, jól kivehető határvonalakkal rendelkeznek)

9. Eredményül a Python output ablakában az eredeti képre ráhelyezett maszkon zölddel besatírozva találhatók az azonsított épületek.



=====================================



III. A script használata:

1. Nyissuk meg az init.py scriptet

2. Adjuk meg az input ill. output fájlok lokális elérési útját a fájl első soraiban definiált változóban, pl.:

input_img = 'C:\\Users\\micro\\Desktop\\Gepilatas\\input\\07.png'

(Ügyeljünk az esetleges escape-karakterekre, pl. backslash)

3. Szükség esetén módosítsuk a képátméretezés, színek ill. tresholdok értékét (ld. 2/5. pont)

4. Futtassuk a scriptet standard módon parancssorból, paraméterek/argumentumok használata nélkül: 

`python init.py`


=====================================


IV. Tesztelés, inputok

A mellékelt input mappában több léptékű és forrású műholdkép-részlet adott.

A várt eredményektől függően további beállítási lehetőség nyílik:

- a zöldterületek RGB-tartományának meghatározásához ("mennyire zöld" a terület, amit ignorálni szeretnénk)
- bináris treshold esetén, szürkeárnyalatban mely sötét ill. világos tartományokat szeretnénk figyelmen kívül hagyni
- kép kicsinyítésének mértéke / target pixel x pixel képméret (arányos a képfeldolgozás ill. kontúrok generálásának sebességével, értelemszerűen)
- körbezárt kontúrok minimum ill maximum kerülete, területe is szabályozható (ld. refineContours függvény)


=====================================


V. Ismert korlátok:

- zöldtetős megoldás esetén az épületek felismerése kétséges.
- világos burkolatú utak, parkolók időnként fals pozitív értékeket adnak
- fekete cseréppel fedett tetők felismerése hiányos
- erősen szabálytalan alakú épületek felismerése elváráson aluli. (a Győri Nemzeti Színház ívelt épületét a próbált konfigurációkban pl. felismerjük, más esetekben hibák adódnak)



