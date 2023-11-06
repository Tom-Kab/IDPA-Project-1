indexMapping = {
    "properties":{
        "product_id":{
            "type":"long"
        },
        "link":{
            "type":"text"
        },
        "size":{
            "type":"text"
        },
        "variant_sku":{
            "type":"text"
        },
        "brand":{
            "type":"text"
        },
        "care_instructions":{
            "type":"text"
        },
        "dominant_material":{
            "type":"text"
        },
        "title":{
            "type":"text"
        },
        "actual_color":{
            "type":"text"
        },
        "dominant_color":{
            "type":"text"
        },
        "product_type":{
            "type":"text"
        },
        "images":{
            "type":"text"
        },
        "body":{
            "type":"text"
        },
        "product_details":{
            "type":"text"
        },
        "size_fit":{
            "type":"text"
        },
        "complete_the_look":{
            "type":"text"
        },
        "type":{
            "type":"text"
        },
        "variant_price":{
            "type":"long"
        },
        "variant_compare_at_price":{
            "type":"long"
        },
        "ideal_for":{
            "type":"text"
        },
        "is_in_stock":{
            "type":"text"
        },
        "inventory":{
            "type":"long"
        },
        "specifications":{
            "type":"text"
        },
        "DetailsVector":{
            "type":"dense_vector",
            "dims":768,
            "index":True,
            "similarity":"l2_norm"
        }
    }
}