# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class SpiderCrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    movie = scrapy.Field()
    year = scrapy.Field()
    runtime = scrapy.Field()
    genres = scrapy.Field()
    directors = scrapy.Field()
    stars = scrapy.Field()
    link = scrapy.Field()
