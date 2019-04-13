# -*- coding: utf-8 -*-
import scrapy
from ..items import SpiderCrawlerItem

class TmdbSpider(scrapy.Spider):
    name = 'tmdbp'

    start_urls = []
    for page_num in range(1, 500):
        start_urls.append("https://www.themoviedb.org/genre/878/movie?page={page_num}".format(page_num=page_num))
    #start_urls = ['https://www.themoviedb.org/genre/878/movie?page=1']

    def parse(self, response):
        items = SpiderCrawlerItem()

        # Extracting the content using xpath selectors
        relative_link = response.xpath('//div[@class="title"]//a/@href').extract()
        html_link = ['https://www.themoviedb.org' + _ for _ in relative_link]

        # create a dictionary to store the scraped info
        for a,b in enumerate(html_link):

            items['movie'] = a
            items['link'] = b


            yield items

