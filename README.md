# Proje 2 (Orta): Makine Arıza Tahmini ve Maliyete Dayalı Bakım Kararı

60 makinenin 365 günlük sensör verisinden (titreşim, sıcaklık, basınç, yük) **önümüzdeki 7 gün içinde arıza olacak mı?** sorusunu tahmin eder ve modeli doğruluk yerine **işletme maliyetiyle** değerlendirir.

## Neyi gösteriyor?

1. **Zaman serisi özellik mühendisliği:** onarımda sıfırlanan kayan pencereler (3 ve 7 gün ortalama, 7 gün std, kısa-uzun ortalama farkı) ve makine yaşı.
2. **Sızıntısız değerlendirme:** rastgele bölme yerine zaman bazlı bölme ve genişleyen pencereli çapraz doğrulama. Eğitim ile doğrulama arasında `HORIZON` günlük boşluk bırakılır, çünkü etiketler ileriye bakar.
3. **Dengesiz sınıf:** pozitif oranı yaklaşık %3.7. Bu yüzden başarı ölçütü doğruluk (accuracy) değil, PR-AUC'dir. Modellerde sınıf ağırlıkları kullanılır.
4. **Model karşılaştırması:** Lojistik Regresyon, Random Forest, Gradient Boosting.
5. **Maliyete dayalı eşik:** 0.50 varsayılan eşiği yerine, alarm ve kaçırılan arıza maliyetini en aza indiren eşik seçilir. Eşik yalnızca eğitim dönemi out-of-fold tahminlerinden seçilir, test dönemine bakılmaz.

## Çalıştırma

```bash
pip install -r requirements.txt
python -m src.run        # veriyi üretir, modelleri eğitir, outputs/ klasörüne yazar (~1 dk)
python -m pytest -q      # testler
```

## Maliyet varsayımı

Karar birimi bir **makine-gündür**. Plansız arıza ≈ 10.000 birim maliyetlidir. Bir arıza penceresi yaklaşık 7 pozitif satıra denk geldiği için kaçırılan her pozitif satır ≈ 1.430 sayılır. Her alarm (denetim veya önleyici bakım) 300 birimdir. Değerler `src/modeling.py` içinde `COST_MISSED` ve `COST_ALARM` olarak tanımlıdır, kendi işletmenize göre değiştirin.

Basitleştirme: aynı arıza için ardışık günlerde verilen alarmlar ayrı ayrı sayılır. Gerçek uygulamada bir makineye tek denetim yeterli olabilir. Bu, "alarm yorgunluğu" ve olay bazlı maliyet için geliştirme fırsatıdır.

## Örnek sonuç (seed=7)

Test dönemi (gün 265 ve sonrası, 5.543 makine-gün, 251 pozitif):

| Strateji | Alarm | Recall | Precision | Toplam maliyet |
|---|---|---|---|---|
| Bakım yok | 0 | 0.00 | – | 358.930 |
| Her gün alarm | 5.543 | 1.00 | 0.05 | 1.662.900 |
| Model, eşik 0.50 | 270 | 0.68 | 0.63 | 195.400 |
| **Model, maliyet-optimal eşik (0.32)** | 362 | 0.78 | 0.54 | **188.680** |

- Seçilen model Random Forest (CV PR-AUC 0.52). Lojistik Regresyon 0.44, Gradient Boosting 0.51 aldı.
- Model, "bakım yok" stratejisine göre maliyeti yaklaşık **%47** düşürüyor.
- En önemli özellikler titreşim (anlık ve kayan ortalama) ve makine yaşı.
- Test PR-AUC'nin CV PR-AUC'den yüksek çıkması beklenen bir durumdur: CV katlarında model daha az veriyle (özellikle ilk katlarda) eğitilir, test için tüm eğitim verisi kullanılır.
- Veri sentetik olduğu için sayıların kendisi değil, **yöntemin akışı** önemlidir. Kendi verinizde sonuçlar farklı çıkacaktır.

## Çıktılar (`outputs/`)

`model_comparison.csv`, `strategy_costs.csv`, `feature_importance.csv`, `sensor_example.png`, `pr_curves.png`, `threshold_cost.png`, `feature_importance.png`, `strategy_costs.png`

## Kendi verinizle kullanma

`data/sensor_data.csv` dosyasını şu sütunlarla değiştirin: `machine_id, day, vibration, temperature, pressure, load, failure` (arıza günü = 1). Dosya varsa üretim atlanır. Farklı sensörler için `src/features.py` içindeki `SENSORS` listesini güncelleyin.

## Geliştirme fikirleri

- Hayatta kalma analizi (survival analysis) ile kalan faydalı ömür (RUL) tahmini.
- SHAP ile bireysel alarm açıklamaları.
- Olay bazlı maliyet: bir arıza için ilk doğru alarmdan sonraki alarmları tekrar saymama.
