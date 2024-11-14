# -*- coding: utf-8 -*-
"""
This defines the Pydantic model to handle incoming image URLs.
"""

from pydantic import BaseModel

class DiseaseImage(BaseModel):
    image: str 
