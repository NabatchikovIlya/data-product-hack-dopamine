col_mapper = {
    1: 'Время от времени я мечтаю/фантазирую о возможных событиях в моей жизни.',
    2: 'Я часто испытываю теплые чувства, заботу по отношению к тем, кто менее успешен чем я.',
    3: 'Мне бывает трудно поставить себя на место другого человека.',
    4: 'Я не всегда испытываю сочувствие к людям, находящихся в трудной жизненной ситуации.',
    5: 'Я сильно сопереживаю персонажам фильмов/героям книг.',
    6: 'В стрессовых ситуациях я испытываю тревогу и дискомфорт.',
    7: 'Обычно я смотрю фильм или спектакль довольно отстраненно, и крайне редко то, что происходит на экране или на сцене, захватывает меня.',
    8: 'При принятии решения я расcматриваю проблему с разных сторон.',
    9: 'Когда я вижу, что человека используют или обманывают, я чувствую, что хочу защитить/поддержать его.',
    10: 'Я чувствую себя беспомощно в эмоционально напряженной, стрессовой ситуации.',
    11: 'Иногда, для того чтобы лучше понять своих друзей, я смотрю на ситуацию с их точки зрения.',
    12: 'В процессе просмотра фильма/чтения книги я редко погружаюсь в происходящее на экране/книге.',
    13: 'Если я являюсь свидетелем ситуации, в которой человек испытывает боль/страдания, то я остаюсь спокойным.',
    14: 'Неудачи других людей обычно не очень сильно волнуют меня.',
    15: 'Если я уверен/уверена в своей правоте, я не буду тратить много времени на то, чтобы выслушать аргументы других людей.',
    16: 'После посещения кино/театра я представляю себя одним из персонажей.',
    17: 'Меня пугает возможность оказаться в стрессовой ситуации.',
    18: 'Иногда, когда я становлюсь свидетелем ситуации несправедливого отношения к человеку, это не вызывает у меня переживаний.',
    19: 'Я достаточно хорошо справляюсь со стрессовыми ситуациями.',
    20: 'Меня достаточно легко растрогать.',
    21: 'Я считаю что у ситуации есть две стороны и всегда стараюсь посмотреть на проблему с разных сторон.',
    22: 'Я считаю себя довольно мягкосердечным человеком.',
    23: 'Во время просмотра фильма/сериала я с легкостью представляю себя на месте главного героя.',
    24: 'В стрессовой ситуации обычно чувствую себя потерянным.',
    25: 'Когда я злюсь или чувствую себя расстроенным, обычно стараюсь посмотреть на случившееся глазами собеседника.',
    26: 'Когда я читаю интересную книгу, я представляю, какие эмоции я бы испытывал находясь на месте героев этого произведения.',
    27: 'Когда я вижу, что кто-то срочно нуждается в помощи в критической ситуации, я теряюсь и не могу оказать из-за этого достойную поддержку.',
    28: 'Перед тем, как критиковать кого-либо, я стараюсь представить, как бы я чувствовал себя на его месте.',
    29: 'Я имею высокое чувство собственного достоинства.',
    30: 'Вам постоянна нужна внешняя мотивация для деятельности, вы склонны испытывать скуку.',
    31: 'У меня был определенный период жизни, когда я получал предупреждения от органов правопорядка, но в последствии снова нарушал.',
    32: 'Вы не испытываете чувство вины или раскаяния.',
    33: 'Вы часто лжете, даже в тех ситуациях где это вам не нужно.',
    34: 'Вы являетесь неэмоциональным человеком.',
    35: 'Я нарушал закон в разных областях правонарушений.',
    36: 'Я имел опыт нарушать закон, когда был несовершеннолетним.',
    37: 'Я часто обманываю и манипулирую людьми.',
    38: 'Я часто обвиняю других в своих проблемах и не хочу нести ответственность за свои поступки.',
    39: 'Бывали случаи, когда я проявлял жесткость по отношению к другим людям.',
    40: 'У меня было много серьезных отношений, которые распадались за короткое время.',
    41: 'У меня нет долгосрочных планов/плыву по течению.',
    42: 'Я часто использую людей в своих целях.',
    43: 'Я часто совершаю необдуманные действия под воздействием эмоций/являюсь импульсивным человеком.',
    44: 'В социуме я виду себя непринужденно и легко.',
    45: 'Я некоторое время имел опыт случайных половых связей.',
    46: 'Мне трудно контролировать свое поведение и эмоциональные импульсы.',
    47: 'Я жестокий человек, не испытываю сочувствия к другим людям.',
    48: 'Я часто избегаю ответственности и каких-либо обязательств в жизни.',
}

score_ind = [1, 2, 5, 6, 8, 9, 10, 11, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28]
reversed_score_ind = [3, 4, 7, 12, 13, 14, 15, 18, 19]
psycho_score_ind = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
fs_ind = [1, 5, 7, 12, 16,  23, 26]
es_ind = [2, 4, 9, 14, 18, 20, 22]
pt_ind = [3, 8, 11, 15, 21, 25, 28]
pd_ind = [6, 10, 13, 17, 19, 24, 27]

score_cols = [col_mapper[ind] for ind in score_ind]
reversed_score_cols = [col_mapper[ind] for ind in reversed_score_ind]
psycho_score_cols = [col_mapper[ind] for ind in psycho_score_ind]
fs_cols = [col_mapper[ind] for ind in fs_ind]
es_cols = [col_mapper[ind] for ind in es_ind]
pt_cols = [col_mapper[ind] for ind in pt_ind]
pd_cols = [col_mapper[ind] for ind in pd_ind]


score_mapper = {
    'Полностью не согласна/согласен': .0,
    'Полность не согласна/согласен': .0,
    'Частично не согласна/согласен': 1.0,
    'Затрудняюсь сказать': 2.0,
    'Частично согласна/согласен': 3.0,
    'Совершенно согласна/согласен': 4.0
}

psycho_score_mapper = {
    'Полностью не согласна/согласен': .0,
    'Полность не согласна/согласен': .0,
    'Частично не согласна/согласен': 0.5,
    'Затрудняюсь сказать': 1.0,
    'Частично согласна/согласен': 1.5,
    'Совершенно согласна/согласен': 2.0
}

reversed_score_mapper = {
    'Полностью не согласна/согласен': 4.0,
    'Полность не согласна/согласен': 4.0,
    'Частично не согласна/согласен': 3.0,
    'Затрудняюсь сказать': 2.0,
    'Частично согласна/согласен': 1.0,
    'Совершенно согласна/согласен': .0
}
