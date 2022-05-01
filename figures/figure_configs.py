configs_fig1 = {
    # StyleGAN2 cars W
    'Redness':          (22,  9, 11,   -8, False),
    'Horizontal flip':  ( 0,  0,  5,  2.0, True),
    'Add grass':        (41,  9, 11,  -18, False),
    'Blocky shape':     (16,  3,  6,   25, False),

    # StyleGAN2 ffhq
    'frizzy_hair':             (31,  2,  6,  20, False),
    'background_blur':         (49,  6,  9,  20, False),
    'bald':                    (21,  2,  5,  20, False),
    'big_smile':               (19,  4,  5,  20, False),
    'caricature_smile':        (26,  3,  8,  13, False),
    'scary_eyes':              (33,  6,  8,  20, False),
    'curly_hair':              (47,  3,  6,  20, False),
    'dark_bg_shiny_hair':      (13,  8,  9,  20, False),
    'dark_hair_and_light_pos': (14,  8,  9,  20, False),
    'dark_hair':               (16,  8,  9,  20, False),
    'disgusted':               (43,  6,  8, -30, False),
    'displeased':              (36,  4,  7,  20, False),
    'eye_openness':            (54,  7,  8,  20, False),
    'eye_wrinkles':            (28,  6,  8,  20, False),
    'eyebrow_thickness':       (37,  8,  9,  20, False),
    'face_roundness':          (37,  0,  5,  20, False),
    'fearful_eyes':            (54,  4, 10,  20, False),
    'hairline':                (21,  4,  5, -20, False),
    'happy_frizzy_hair':       (30,  0,  8,  20, False),
    'happy_elderly_lady':      (27,  4,  7,  20, False),
    'head_angle_up':           (11,  1,  4,  20, False),
    'huge_grin':               (28,  4,  6,  20, False),
    'in_awe':                  (23,  3,  6, -15, False),
    'wide_smile':              (23,  3,  6,  20, False),
    'large_jaw':               (22,  3,  6,  20, False),
    'light_lr':                (15,  8,  9,  10, False),
    'lipstick_and_age':        (34,  6, 11,  20, False),
    'lipstick':                (34, 10, 11,  20, False),
    'mascara_vs_beard':        (41,  6,  9,  20, False),
    'nose_length':             (51,  4,  5, -20, False),
    'elderly_woman':           (34,  6,  7,  20, False),
    'overexposed':             (27,  8, 18,  15, False),
    'screaming':               (35,  3,  7, -15, False),
    'short_face':              (32,  2,  6, -20, False),
    'show_front_teeth':        (59,  4,  5,  40, False),
    'smile':                   (46,  4,  5, -20, False),
    'straight_bowl_cut':       (20,  4,  5, -20, False),
    'sunlight_in_face':        (10,  8,  9,  10, False),
    'trimmed_beard':           (58,  7,  9,  20, False),
    'white_hair':              (57,  7, 10, -24, False),
    'wrinkles':                (20,  6,  7, -18, False),
    'boyishness':              (8,   2,  5,  20, False),
}

# Model, layer, edit, layer_start, layer_end, class, sigma, idx, name, (example seeds)
configs_fig7 = [
    # StyleGAN2 cars

    # In paper
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'car', 20.0, 50, 'Autumn', [329004386]),
    ('StyleGAN2', 'style', 'latent', 'w', 0, 4, 'car', -10, 15, 'Focal length', [587218105, 361309542, 1355448359]),
    ('StyleGAN2', 'style', 'latent', 'w', 0, 9, 'car', 18.5, 44, 'Car model', [1204444821]),
    ('StyleGAN2', 'style', 'latent', 'w', 7, 9, 'car', 20.0, 18, 'Reflections', [1498448887]),

    # Other
    ('StyleGAN2', 'style', 'latent', 'w', 9, 11, 'car', -20.0, 41, 'Add grass', [257249032]),
    ('StyleGAN2', 'style', 'latent', 'w', 0, 5, 'car', -2.7, 0, 'Horizontal flip', [1221001524]),
    ('StyleGAN2', 'style', 'latent', 'w', 7, 16, 'car', 20.0, 50, 'Fall foliage', [1108802786]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'car', -14.0, 29, 'Blown out highlight', [490151100, 1010645708]),
    ('StyleGAN2', 'style', 'latent', 'w', 0, 4, 'car', 12, 13, 'Flat vs tall', [1541814754, 1355448359]),
    ('StyleGAN2', 'style', 'latent', 'w', 5, 6, 'car', 20.0, 32, 'Front wheel turn', [1060866846]),
    ('StyleGAN2', 'style', 'latent', 'w', 9, 10, 'car', -20.0, 35, 'Ground smoothness', [1920211941]),
    ('StyleGAN2', 'style', 'latent', 'w', 7, 16, 'car', 20.0, 37, 'Image contrast', [1419881462]),
    ('StyleGAN2', 'style', 'latent', 'w', 9, 11, 'car', -20.0, 45, 'Sepia', [105288903]),
    ('StyleGAN2', 'style', 'latent', 'w', 7, 16, 'car', 20.0, 38, 'Sunset', [1419881462]),
    ('StyleGAN2', 'style', 'latent', 'w', 0, 5, 'car', -2.0, 1, 'Side to front', [1221001524]),
    ('StyleGAN2', 'style', 'latent', 'w', 3, 7, 'car', -7.5, 10, 'Sports car', [743765988]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'car', 5.3, 14, 'White car', [1355448359]),


    # StyleGAN2 ffhq

    # In paper
    ('StyleGAN2', 'style', 'latent', 'w', 6, 8, 'ffhq', -20.0, 43, 'Disgusted', [140658858, 1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', 9.0, 0, 'Makeup', [266415229]), #, 375122892]),

    # Other
    ('StyleGAN2', 'style', 'latent', 'w', 4, 5, 'ffhq', 10.0, 19, 'Big smile', [427229260]),
    ('StyleGAN2', 'style', 'latent', 'w', 6, 8, 'ffhq', -20.0, 33, 'Scary eyes', [1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 2, 5, 'ffhq', 18.2, 21, 'Bald', [1635892780]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', 13.0, 13, 'Bright BG vs FG', [798602383]),
    ('StyleGAN2', 'style', 'latent', 'w', 3, 6, 'ffhq', -60.0, 47, 'Curly hair', [1140578688]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', -10.2, 16, 'Hair albedo', [427229260]),
    ('StyleGAN2', 'style', 'latent', 'w', 4, 7, 'ffhq', 10.0, 36, 'Displeased', [1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', 20.0, 37, 'Eyebrow thickness', [1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 7, 8, 'ffhq', -30.0, 54, 'Eye openness', [11573701]),
    ('StyleGAN2', 'style', 'latent', 'w', 0, 5, 'ffhq', 20.0, 37, 'Face roundness', [1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 4, 10, 'ffhq', -20.0, 54, 'Fearful eyes', [11573701]),
    ('StyleGAN2', 'style', 'latent', 'w', 4, 5, 'ffhq', -13.6, 21, 'Hairline', [1635892780]),
    ('StyleGAN2', 'style', 'latent', 'w', 0, 8, 'ffhq', 20.0, 30, 'Happy frizzy hair', [1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 1, 4, 'ffhq', -10.5, 11, 'Head angle up', [798602383]),
    ('StyleGAN2', 'style', 'latent', 'w', 3, 6, 'ffhq', -15.0, 23, 'In awe', [1635892780]),
    ('StyleGAN2', 'style', 'latent', 'w', 3, 6, 'ffhq', -15.0, 22, 'Large jaw', [1635892780]),
    ('StyleGAN2', 'style', 'latent', 'w', 10, 11, 'ffhq', 20.0, 34, 'Lipstick', [1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 4, 5, 'ffhq', -30.0, 51, 'Nose length', [11573701]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 18, 'ffhq', 5.0, 27, 'Overexposed', [1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 3, 7, 'ffhq', -14.5, 35, 'Screaming', [1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 2, 6, 'ffhq', -20.0, 32, 'Short face', [1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 4, 5, 'ffhq', -20.0, 46, 'Smile', [1175071341]),
    ('StyleGAN2', 'style', 'latent', 'w', 4, 5, 'ffhq', -20.0, 20, 'Unhappy bowl cut', [1635892780]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', -8.0, 10, 'Sunlight in face', [798602383]),
    ('StyleGAN2', 'style', 'latent', 'w', 7, 9, 'ffhq', -40.0, 58, 'Trimmed beard', [1602858467]),
    ('StyleGAN2', 'style', 'latent', 'w', 3, 5, 'ffhq', -9.0, 20, 'Forehead hair', [1382206226]),
    ('StyleGAN2', 'style', 'latent', 'w', 0, 5, 'ffhq', -9.0, 21, 'Happy frizzy hair', [1382206226]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', -15.0, 25, 'Light UD', [1382206226]),
    # ('StyleGAN2', 'style', 'latent', 'w', 8, 11, 'ffhq', 9.0, 0, 'Makeup', [1953272274]),
    ('StyleGAN2', 'style', 'latent', 'w', 4, 6, 'ffhq', -16.0, 36, 'Smile', [1382206226]),

    # Custom
    ('StyleGAN2', 'style', 'latent', 'w', 0, 15, 'ffhq', -10, 69, 'Human hair color', [1, 452352]),

    # StyleGAN2 horse

    # In paper
    ('StyleGAN2', 'style', 'latent', 'w', 3, 5, 'horse', -2.9, 3, 'Add rider', [944988831]),
    ('StyleGAN2', 'style', 'latent', 'w', 5, 7, 'horse', -7.8, 11, 'Coloring', [897830797]),

    # Other
    ('StyleGAN2', 'style', 'latent', 'w', 7, 9, 'horse', 11.8, 20, 'White horse', [1042666993]),
    ('StyleGAN2', 'style', 'latent', 'w', 9, 11, 'horse', 9.0, 8, 'Green bg', [897830797]),


    # StyleGAN2 cat

    # In paper
    ('StyleGAN2', 'style', 'latent', 'w', 5, 8, 'cat', 20.0, 45, 'Eyes closed', [81011138]),
    ('StyleGAN2', 'style', 'latent', 'w', 2, 5, 'cat', 20.0, 27, 'Fluffiness', [740196857]),

    # Other
    ('StyleGAN2', 'style', 'latent', 'w', 0, 6, 'cat', 20.0, 18, 'Head dist 2', [2021386866]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'cat', 12.7, 28, 'Light pos', [740196857]),


    # StyleGAN2 church

    # In paper
    ('StyleGAN2', 'style', 'latent', 'w', 7, 9, 'church', -20.0, 20, 'Clouds', [1360331956, 485108354]),
    ('StyleGAN2', 'style', 'latent', 'w', 7, 9, 'church', -8.4, 8, 'Direct sunlight', [1777321344, 38689046]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'church', 20.0, 15, 'Sun direction', [485108354]),
    ('StyleGAN2', 'style', 'latent', 'w', 12, 14, 'church', -20.0, 8, 'Vibrant', [373098621, 38689046]),

    # Other
    ('StyleGAN2', 'style', 'latent', 'w', 9, 14, 'church', 9.9, 11, 'Blue skies', [1003401116]),
    ('StyleGAN2', 'style', 'latent', 'w', 5, 7, 'church', -20.0, 20, 'Clouds 2', [1360331956, 485108354]),
    ('StyleGAN2', 'style', 'latent', 'w', 5, 6, 'church', -19.1, 12, 'Trees', [1344303167]),


    # StyleGAN1 bedrooms

    # In paper
    ('StyleGAN', 'g_mapping', 'latent', 'w', 0, 6, 'bedrooms', 18.5, 31, 'flat_vs_tall', [2073683729]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 0, 3, 'bedrooms', -2.6, 5, 'Bed pose', [96357868]),


    # StyleGAN1 wikiart

    # In paper
    ('StyleGAN', 'g_mapping', 'latent', 'w', 0, 2, 'wikiart', -2.9, 7, 'Head rotation', [1819967864]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 8, 15, 'wikiart', 7.5, 9, 'Simple strokes', [1239190942]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 9, 15, 'wikiart', -20.0, 59, 'Skin tone', [1615931059, 1719766582]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 4, 7, 'wikiart', 20.0, 36, 'Mouth shape', [333293845]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 2, 4, 'wikiart', -35.0, 35, 'Eye spacing', [1213732031, 333293856]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 8, 15, 'wikiart', 20.0, 31, 'Sharpness', [1489906162, 1768450051]),

    # Other
    ('StyleGAN', 'g_mapping', 'latent', 'w', 4, 7, 'wikiart', -16.3, 25, 'Open mouth', [1655670048]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 10, 16, 'wikiart', -20.0, 18, 'Rough strokes', [1942295817]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 1, 4, 'wikiart', -7.2, 14, 'Camera UD', [1136416437]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 8, 14, 'wikiart', -8.4, 13, 'Stroke contrast', [1136416437]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 4, 7, 'wikiart', 20.0, 44, 'Eye size', [333293845]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 4, 8, 'wikiart', 13.9, 16, 'Open mouth', [2135985383]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 10, 15, 'wikiart', 20.0, 26, 'Sharpness 2', [1489906162, 1223183477]),
    ('StyleGAN', 'g_mapping', 'latent', 'w', 9, 14, 'wikiart', 20.0, 32, 'Splotchy', [1768450051]),

    # StyleGAN2 ukiyoe
    ('StyleGAN2', 'style', 'latent', 'w', 0, 10, 'ukiyoe', -1.5, 3, 'Face Rotation', [1819967864]),


    # StyleGAN2 beetle
    ('StyleGAN2', 'style', 'latent', 'w', 0, 17, 'beetles', -2.5, 2, 'Pattern', [1, 1819967864]),


    # StyleGAN2 anime
    ('StyleGAN2', 'style', 'latent', 'w', 0, 15, 'anime', -10, 51, 'Laugh', [263284, 123423, 452352, 532745]),
    ('StyleGAN2', 'style', 'latent', 'w', 0, 15, 'anime', -10, 57, 'Eye color', [1]),
    ('StyleGAN2', 'style', 'latent', 'w', 0, 15, 'anime', -10, 69, 'Hair color', [1, 452352]),
]
