import scrapy
import pandas as pd


class LPSE(scrapy.Spider):
  name = 'lpse'
  start_urls = ['https://lpse.kemkes.go.id/eproc4']

  def parse(self, response):
    # print(response.body)
    rows = response.css('div.card.card-primary > table > tbody > tr')
    # print(len(rows))
    
    n = []
    h = []
    a = []
    l = []
    for i in range(1, len(rows)+1):
      jml = response.css('div.card.card-primary > table > tbody > tr:nth-child('+str(i)+') > td.bs-callout-info > span.badge.badge-secondary.float-right::text').extract_first()
      # print(jml)
      # print(type(jml))
      
      if jml is not None and int(jml) > 0:
        new_no = i
        
        for j in range(1, int(jml)+1):
          nama = response.css('div.card.card-primary > table > tbody > tr:nth-child('+str(j + new_no)+') > td:nth-child(2) > a::text').extract_first()
          hps = response.css('div.card.card-primary > table > tbody > tr:nth-child('+str(j + new_no)+') > td.table-hps::text').extract_first().strip()
          akhir_pendaftaran = response.css('div.card.card-primary > table > tbody > tr:nth-child('+str(j + new_no)+') > td.center::text').extract_first()
          link = response.css('div.card.card-primary > table > tbody > tr:nth-child('+str(j + new_no)+') > td > a::attr(href)').extract_first()
          n.append(nama)
          h.append(hps)
          a.append(akhir_pendaftaran)
          l.append('https://lpse.lkpp.go.id'+link)
      # else:
      #   print('not pass')
      # print("==============")


    data = pd.DataFrame()
    data['nama pekerjaan'] = n
    data['HPS'] = h
    data['akhir pendaftaran'] = a
    data['link'] = l
    data.to_excel("lpse_lkpp.xlsx")