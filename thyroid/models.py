from django.db import models

# Create your models here.
class ThyroidImage(models.Model):
    path = models.TextField(max_length=1000)
    label = models.IntegerField()
    crc32 = models.CharField(max_length=1000)

    class Meta:
        db_table = 'thyroid_image'  # 指明数据库表名
