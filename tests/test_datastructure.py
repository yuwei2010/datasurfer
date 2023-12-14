#!/usr/bin/env python

"""Tests for `datastructure` package."""

import pytest

import sys

sys.path.insert(0, r'C:\95_Programming\10_Data_Related\20_Projects\10_Git\10_Datastructure')
import datastructure as ds


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response: None):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


print(dir(ds.dataobjects))