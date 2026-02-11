# DIC MVP Prototipi

Bu proje ardışık çekme testi görüntülerinden piksel tabanlı yer değiştirme/hız alanı üreten basit bir DIC prototipidir.

## Dosyalar

- `dic_app.py`: Uçtan uca uygulama akışı (yükleme, ROI, DIC, çıktı).
- `dic_core.py`: Korelasyon, ROI/grid ve gerinim hesapları.
- `io_utils.py`: Görüntü yükleme ve çıktı kaydetme yardımcıları.
- `viz.py`: Vektör alanı görselleştirme.
- `config.yaml`: Varsayılan parametreler.

## Kurulum

```bash
pip install opencv-python numpy matplotlib pyyaml
```

## Çalıştırma

### 1) Interaktif ROI seçimi ile

```bash
python dic_app.py --input ./images --output ./outputs
```

### 2) Interaktif olmayan ROI ile

```bash
python dic_app.py --input ./images --output ./outputs --roi 100 80 400 300
```

## Çıktılar

Her frame çifti için:

- `pair_XXXX.csv`: `x,y,u,v,vx,vy,score`
- `pair_XXXX_quiver.png`: Görüntü üstünde displacement vektörleri
- `pair_XXXX_strain_*.npy`: `exx`, `eyy`, `gxy`

## Notlar

- `subset_size` tek sayı olmalıdır (örn. 21, 31).
- ROI, `subset_size` ve `step` değerlerine göre otomatik snap edilir.
- `score < min_score` noktaları geçersiz sayılır (`NaN`).
