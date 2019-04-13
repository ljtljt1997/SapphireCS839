# -*- coding: utf-8 -*-
import scrapy
from ..items import SpiderCrawlerItem


class AmazonSpider(scrapy.Spider):
    name = 'tmdb'
    with open('D://Projects//839_EM//spider_crawler//links.txt') as f:
        start_urls = [url.strip() for url in f.readlines()]

    def parse(self, response):
        items = SpiderCrawlerItem()

        title = response.css('title::text')
        movie_name = title.re('(.*)\([0-9]{4}\).*')
        release_year = title.re('.*\(([0-9]{4})\).*')
        runtime = response.xpath('//section//p[contains(., "Runtime")]/text()').extract()
        genres = ', '.join(response.xpath('//section[@class="genres right_column"]//ul//li//a/text()').extract())
        stars = ', '.join(response.xpath('//section[@class="panel top_billed scroller"]//ol//li//p//a/text()').extract())

        name_tmp = []
        for people in response.xpath('//ol[@class = "people no_image"]//li'):
            identity = people.xpath('.//p[@class = "character"]/text()').extract()[0]
            if 'Director' in identity:
                name_tmp.extend(people.xpath('.//p//a/text()').extract())

        directors = ', '.join(name_tmp)


        for item in zip(movie_name, release_year, runtime, [genres], [directors], [stars]):
            items['movie'] = item[0]
            items['year'] = item[1]
            items['runtime'] = item[2]
            items['genres'] = item[3]
            items['directors'] = item[4]
            items['stars'] = item[5]

            yield items