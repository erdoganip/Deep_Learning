Kod bu haliyle direkt olarak CIFAR-10 datasetini indirip çalışmaya başlar.
36. satırdan batch size'ı, 37. satırdan epoch sayısını değiştirebilirsiniz. 
Kodun çalışma aşamaları aşağıdaki gibidir:
1-Dataseti indir, transform fonksiyonlarını uygula, dataloaderları hazırla
2-Modeli, Loss fonksiyonunu ve Optimizer'ı tanımla
3-Belirtilen epoch sayısı içerisinde modeli eğitmeye başla
  -> Her epoch'un sonunda eval() fonksiyonu çağrılır. Test datası ile model test edilir.
  -> 0.,20.,40. ve 49.(son) epochlarda model kaydedilir, latent representationlar alınıp visualize edilir (tSNE). Burası zaman aldığı için kodu hızlı çalıştırmak isterseniz 84-97. Satırlar arasını commentleyip etkisizleştirebilirsiniz. Kod train ve testing işlemlerini tamamlayacak, yalnızca tSNE plotting kısmını yapmayacak ve model weightlerini kaydetmeyecektir.

4-Deney bitince Training ve Testing Accuracy/Epochs ve Training ve Testing Loss/Epochs grafiklerini çiz

