# -*- coding: utf-8 -*-
import scrapy
from ..items import SpiderCrawlerItem

class ImdbSpiderSpider(scrapy.Spider):
    name = 'imdb'
    start_urls = ["https://www.imdb.com/search/title?genres=sci-fi&start={start}&explore=title_typegenres&title_type=movie&ref_=adv_explore_rhs".format(start=start)
                  for start in range(1, 6001, 50)]

    def parse(self, response):
        items = SpiderCrawlerItem()

        # Extract information from the web page.
        movie_names, release_year, genres, runtime, directors, stars = [], [], [], [], [], []
        for item in response.xpath('//div[@class = "lister-item-content"]'):
            movie_names.extend(item.xpath('.//h3//a[contains(@href, "title")]/text()').extract() if item.xpath(
                './/h3//a[contains(@href, "title")]/text()').extract() else [None])
            release_year.extend(item.xpath('.//h3//span[@class = "lister-item-year text-muted unbold"]/text()').re(
                '([0-9]{4})') if item.xpath('.//h3//span[@class = "lister-item-year text-muted unbold"]/text()').re(
                '([0-9]{4})') else [None])
            genres.extend(item.xpath('.//span[@class = "genre"]/text()').re(r'\n(.*?)[ ]+$') if item.xpath(
                './/span[@class = "genre"]/text()').re(r'\n(.*?)[ ]+$') else [None])
            runtime.extend(item.xpath('.//span[@class = "runtime"]/text()').extract() if item.xpath(
                './/span[@class = "runtime"]/text()').extract() else [None])

            flag1 = item.xpath('.//p[contains(text(),"Director")]')
            flag2 = item.xpath('.//p[contains(text(),"Stars")]')
            if not flag1:
                directors.append(None)
                if not flag2:
                    stars.append(None)
                else:
                    stars.append(', '.join(flag2.xpath('./*/node()').extract()))
            else:
                pp = [[], []]
                flag = 0
                for tmp in flag1.xpath('./*/node()').extract():
                    if tmp == '|':
                        flag = 1
                        continue
                    pp[flag].append(tmp)

                directors.append(', '.join(pp[0]))
                stars.append(', '.join(pp[1]) if flag else None)


        #Store data into items
        for item in zip(movie_names, release_year, runtime, genres, directors, stars):
            # create a dictionary to store the scraped info
            items['movie'] = item[0]
            items['year'] = item[1]
            items['runtime'] = item[2]
            items['genres'] = item[3]
            items['directors'] = item[4]
            items['stars'] = item[5]

            yield items
