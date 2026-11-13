#!/usr/bin/env python3
"""Round 3 of the generated training corpus (2026-11-13).

Round 3 teaches what candidate 2 missed or mislabeled on the blind gold
batches test5 to test7 (training/README.md records each run):

- PERSON: hyphenated surnames, particles, accents, ALL-CAPS form values,
  initials, names behind a title, bare first names in greetings, sign-offs
  and prose, a first name or surname that returns after the full name, chat
  speakers, "Surname, First" rows;
- ADDRESS: UK lines with a flat or a house name and a postcode, and the
  German, Spanish, Italian, Portuguese, Polish, Dutch and Irish orders;
- DATE_OF_BIRTH: ISO, dashed, abbreviated-month and block-capital values
  under English and foreign birth labels;
- SENSITIVE_ORGANIZATION: the facility kinds of the labeling guide, in mixed
  case, in block capitals and with a leading Romance or German word;
- negatives: letterheads, companies under role labels, employers, products
  with a digit, quantities, ordinary dates in every birth-date format, a
  year in front of a title, cities with a site word, role and department
  addressees, empty form fields, sentences and field values that open with
  an ordinary capitalised word, named public places, bank codes, a
  nationality in front of an identity document, "City of X" and regions.

Three training runs shaped it. The first candidate over-redacted on the
older gold sets (label words, sentence openers, products as addresses), so
the negatives above grew; the second lost streets with no street word; and
the private regression documents showed official records in the Russian
order (surname first, patronymics, initials, postcode ahead of the street),
which the last genres cover with invented names.

It deliberately does NOT draw on the test8 blind run (French documents,
table cells, products under a bare "Name:", column gaps, address blocks with
the town on its own line, surnames that are company words): test8 is the
held-out measurement for the candidate trained on this corpus.

Unlike rounds 1 and 2, a labeled string may occur more than once in a
document, so every span carries its character offsets. Genre functions build
text with markers (Builder3.P / A / D / O) and render() turns the markers
into offsets.
"""
import re
import string

MARK_OPEN, MARK_SEP, MARK_CLOSE = "", "", ""
SEED_ROUND3 = 20261113

# Pools: every entry was checked against the gold documents and the round-1 and
# round-2 pools on 2026-11-13; an entry sharing a word with either was dropped.
# Train and dev stay disjoint.
# Keep the lists as they are: the generator draws from them by position, so
# removing one entry changes every later document and the corpus no longer
# matches the model trained on it. Add new entries in a new round.
TRAIN_FIRST = ['Chidinma', 'Obinna', 'Uchenna', 'Folasade', 'Babatunde', 'Abena', 'Yaw',
 'Sipho', 'Nomvula', 'Zanele', 'Chipo', 'Wanjiru', 'Otieno', 'Amina', 'Fatou',
 'Aissatou', 'Seun', 'Damilola', 'Ebele', 'Ikenna', 'Titilayo', 'Kelechi',
 'Lindiwe', 'Mpho', 'Kagiso', 'Gwilym', 'Eluned', 'Iestyn', 'Aoife', 'Niamh',
 'Padraig', 'Sorcha', 'Eoghan', 'Fionnuala', 'Ruairi', 'Catriona', 'Hamish',
 'Iona', 'Morven', 'Ailsa', 'Małgorzata', 'Zbigniew', 'Grażyna', 'Jadwiga',
 'Bożena', 'Radek', 'Vlasta', 'Zdeňka', 'Bence', 'Réka', 'Ildikó', 'Dragan',
 'Milica', 'Vesna', 'Zoran', 'Åsa', 'Björn', 'Sigrid', 'Gunnar', 'Annika',
 'Günther', 'Friederike', 'Lieselotte', 'Wilhelmina', 'Maarten', 'Jeroen',
 'Femke', 'Bram', 'Inés', 'Begoña', 'Giulia', 'Alessandro', 'Chiara',
 'Lorenzo', 'Benedetta', 'Tiago', 'Luísa', 'Arjun', 'Deepika', 'Rajesh',
 'Ananya', 'Harpreet', 'Gurpreet', 'Manpreet', 'Farhan', 'Imran', 'Nusrat',
 'Tahmina', 'Sanjay', 'Meera', 'Minh', 'Thanh', 'Huong', 'Linh', 'Xiaoling',
 'Sakura', 'Takeshi', 'Putri', 'Budi', 'Nurul', 'Ayşe', 'Emre', 'Zeynep',
 'Layla', 'Hassan', 'Karim', 'Yasmin', 'Reza', 'Shirin', 'Margaret',
 'Deborah', 'Nigel', 'Lorraine', 'Trevor', 'Beverley', 'Graham', 'Hazel',
 'Stuart', 'Joyce', 'Derek', 'Kenneth', 'Lesley', 'Malcolm', 'Glenys',
 'Russell', 'Losana', 'Aroha', 'Wiremu', 'Ngaire', 'Dorothea', 'Rupert',
 'Gwendolyn', 'Ignatius', 'Perpetua', 'Bartosz', 'Krzysztof', 'Agata',
 'Stanisław', 'Svetlana', 'Yevhen', 'Taras', 'Nikolai', 'Esperanza', 'Rocío',
 'Fabrizio', 'Paolo', 'Nuno', 'Catarina', 'Chinonso', 'Yetunde', 'Efua',
 'Kojo', 'Esi', 'Thulani', 'Bongani', 'Sibusiso', 'Kirsty', 'Fraser',
 'Mhairi', 'Dougal', 'Angharad', 'Carys', 'Siân', 'Tomos', 'Deirdre']
DEV_FIRST = ['Akosua', 'Tendai', 'Ousmane', 'Olumide', 'Naledi', 'Cillian', 'Mairead',
 'Wojciech', 'Łukasz', 'Zoltán', 'Torbjørn', 'Jörg', 'Sanne', 'Montserrat',
 'João', 'Vikram', 'Ayesha', 'Kavita', 'Haruto', 'Mehmet', 'Mariam',
 'Geoffrey', 'Clive', 'Pauline', 'Sione', 'Cornelius', 'Dominika', 'Oksana',
 'Simona', 'Bukola', 'Nokuthula', 'Blodwen', 'Dearbhla']
TRAIN_LAST = ['Balogun', 'Boateng', 'Owusu', 'Appiah', 'Dlamini', 'Ndlovu', 'Mokoena',
 'Moyo', 'Kamau', 'Odhiambo', 'Cissé', 'Pritchard', 'Bevan', 'Griffiths',
 'Cadwallader', 'Donnelly', 'Fitzgerald', 'MacDonald', 'McAllister',
 'Sinclair', 'Buchanan', "O'Sullivan", "O'Rourke", "O'Mahony", 'McGrath',
 'Wiśniewski', 'Szymański', 'Dąbrowska', 'Kaczmarek', 'Procházka', 'Černý',
 'Kovács', 'Szabó', 'Tóth', 'Petrović', 'Marković', 'Popescu', 'Ionescu',
 'Nyström', 'Åkesson', 'Virtanen', 'Korhonen', 'Schröder', 'Müller',
 'Hoffmann', 'Krüger', 'Jansen', 'Hendriks', 'Dekker', 'Rodríguez',
 'Hernández', 'Romero', 'Navarro', 'Iglesias', 'Bianchi', 'Ricci', 'Conti',
 'Oliveira', 'Sharma', 'Patel', 'Reddy', 'Chatterjee', 'Banerjee', 'Singh',
 'Kaur', 'Khan', 'Hussain', 'Ahmed', 'Rahman', 'Chowdhury', 'Nguyen', 'Tran',
 'Pham', 'Wang', 'Zhang', 'Suzuki', 'Yamamoto', 'Santoso', 'Wijaya', 'Yılmaz',
 'Kaya', 'Öztürk', 'Khoury', 'Mansour', 'Nasser', 'Karimi', 'Whitaker',
 'Hargreaves', 'Thornton', 'Fairclough', 'Holloway', 'Blackwood', 'Underhill',
 'Chadwick', 'Atherton', 'Entwistle', 'Ormerod', 'Postlethwaite', 'Aldridge',
 'Bickerstaff', 'Dunleavy', 'Everleigh', 'Goodall', 'Heaton', 'Ingleby',
 'Kettering', 'Lockwood', 'Mawdsley', 'Nethercott', 'Oldroyd', 'Quarmby',
 'Rothwell', 'Stanhope', 'Tattersall', 'Uttley', 'Winterbourne', 'Yardley',
 'Akintola', 'Ogunleye', 'Nkemelu', 'Agyeman', 'Tshabalala', 'Mahlangu',
 'Wanyama', 'Njoroge', 'Sylla', 'Rhydderch', 'Maddocks', "Ó'Dea", 'Mulcahy',
 'Lynch', 'Farquhar', 'Kwiatkowski', 'Pawlak', 'Sikora', 'Pospíšil', 'Balogh',
 'Stojanović', 'Lazăr', 'Dimitrov', 'Dahlgren', 'Nordahl', 'Bjørnstad',
 'Niemi', 'Lehtonen', 'Schumacher', 'Lindemann', 'Bosman', 'Kuiper',
 'Vermeulen', 'Delgado', 'Vargas', 'Serrano', 'Marchetti', 'Caruso', 'Pinto',
 'Sousa', 'Tavares', 'Menon', 'Nair', 'Desai', 'Bhatt', 'Malik', 'Qureshi',
 'Begum', 'Vuong', 'Huang', 'Zhao', 'Kobayashi', 'Watanabe', 'Arslan',
 'Çelik', 'Aydın', 'Farouk', 'Saleh']
DEV_LAST = ['Khumalo', 'Traoré', 'Kavanagh', 'Drummond', 'MacGregor', 'Svoboda',
 'Jovanović', 'Johansen', 'Zimmermann', 'Martínez', 'Esposito', 'Iyer',
 'Gill', 'Siddiqui', 'Chen', 'Demir', 'Hosseini', 'Ashworth', 'Braithwaite',
 'Calloway', 'Jephson', 'Penhaligon', 'Varley', 'Asante', 'Keita', 'Hennessy',
 'Hruška', 'Engström', 'Vogel', 'Castillo', 'Galli', 'Joshi', 'Hoang',
 'Hidayat', 'Rostami']
TRAIN_PARTICLE_LAST = ['van der Meer', 'van Dijk', 'de Groot', 'del Castillo', 'da Costa',
 'di Marco', 'ter Horst', 'van Leeuwen', 'de Wit', 'de Bruijn',
 'della Rovere', 'de Luca', 'da Rocha', 'das Neves', 'von Lindenau',
 'de Graaf', 'van Houten', 'bin Rashid', 'Al-Mansouri', 'El-Sayed',
 'ap Gwilym', 'ap Iorwerth', 'Ní Dhomhnaill', 'Ó Murchú', 'Mac Giolla',
 'du Plessis', 'le Roux', 'van Niekerk', 'van der Merwe', 'de Villiers',
 'Abu Hamdan']
DEV_PARTICLE_LAST = ['dos Reis', 'van Rijn', 'von der Heide', 'Al-Hashimi', 'de Souza',
 'ibn Khalil']
TRAIN_STREET_STEMS = ['Alderley', 'Brackley', 'Cotswold', 'Elmhurst', 'Greenhill', 'Ivybridge',
 'Kingsway', 'Mulberry', 'Sandringham', 'Tollgate', 'Vicarage', 'Wellfield',
 'Yewtree', 'Abbotsford', 'Beechcroft', 'Dovecote', 'Eastgate', 'Fairmead',
 'Gorsebank', 'Heathfield', 'Jesmond', 'Kirkstall', 'Lowther', 'Maplewood',
 'Netherby', 'Quarryside', 'Rookery', 'Sycamore', 'Tanglewood', 'Ullswater',
 'Woodcock', 'Ashdale', 'Blenheim', 'Cranmore', 'Daleside', 'Glebe', 'Hollin',
 'Lynton', 'Moorside', 'Paddock', 'Thornhill', 'Windermere']
DEV_STREET_STEMS = ['Hawthorn', 'Uplands', 'Coppice', 'Inglewood', 'Pinfold', 'Verbena',
 'Ferncliffe', 'Staveley']
TRAIN_UK_TOWNS = ['Skipton', 'Hexham', 'Totnes', 'Malvern', 'Oban', 'Truro', 'Selby',
 'Morpeth', 'Kelso', 'Frome', 'Bodmin', 'Nairn', 'Cromer', 'Otley', 'Ripon',
 'Larne', 'Newry', 'Wick', 'Buxton', 'Matlock', 'Kirkcaldy', 'Dunbar',
 'Stroud', 'Tavistock', 'Wadebridge', 'Louth', 'Sleaford', 'Oakham',
 'Uttoxeter', 'Leominster', 'Welshpool', 'Machynlleth', 'Cardigan',
 'Lampeter', 'Barmouth', 'Forfar', 'Brechin', 'Cupar', 'Peebles', 'Lanark',
 'Ballymena', 'Omagh', 'Enniskillen', 'Clitheroe', 'Settle', 'Haltwhistle',
 'Keighley', 'Ilkley']
DEV_UK_TOWNS = ['Brecon', 'Thirsk', 'Hawick', 'Penrith', 'Helston', 'Bridgnorth',
 'Dolgellau', 'Stranraer', 'Berwick']
TRAIN_DE_TOWNS = ['Regensburg', 'Bamberg', 'Erfurt', 'Tübingen', 'Passau', 'Celle', 'Fulda',
 'Speyer', 'Goslar', 'Bayreuth', 'Konstanz', 'Flensburg']
DEV_DE_TOWNS = ['Görlitz', 'Marburg', 'Wismar']
TRAIN_ES_TOWNS = ['Valencia', 'Zaragoza', 'Granada', 'Murcia', 'Gijón', 'Salamanca', 'Burgos',
 'Cádiz', 'Pamplona', 'Toledo', 'Córdoba']
DEV_ES_TOWNS = ['Vigo', 'Oviedo']
TRAIN_IT_TOWNS = ['Verona', 'Padova', 'Trieste', 'Parma', 'Lecce', 'Bergamo', 'Pisa', 'Modena',
 'Ferrara', 'Siena', 'Brescia', 'Ancona']
DEV_IT_TOWNS = ['Perugia', 'Ravenna']
TRAIN_PT_TOWNS = ['Braga', 'Faro', 'Aveiro', 'Évora', 'Viseu', 'Setúbal', 'Leiria', 'Tomar']
DEV_PT_TOWNS = ['Guimarães']
TRAIN_PL_TOWNS = ['Kraków', 'Wrocław', 'Poznań', 'Łódź', 'Lublin', 'Toruń', 'Opole', 'Rzeszów']
DEV_PL_TOWNS = ['Gdynia', 'Olsztyn']
TRAIN_NL_TOWNS = ['Utrecht', 'Leiden', 'Delft', 'Zwolle', 'Arnhem', 'Deventer', 'Gouda',
 'Alkmaar']
DEV_NL_TOWNS = ['Breda']
TRAIN_IE_TOWNS = ['Ennis', 'Athlone', 'Sligo', 'Kilkenny', 'Mullingar', 'Letterkenny',
 'Dundalk', 'Navan']
DEV_IE_TOWNS = ['Clonmel']
TRAIN_DE_STREETS = ['Lindenstraße', 'Gartenweg', 'Bahnhofstraße', 'Schillerallee', 'Am Mühlbach',
 'Rosenweg', 'Kirchgasse', 'Goetheplatz', 'Feldstraße', 'Uferweg', 'Ahornweg',
 'Brunnenstraße', 'Talstraße']
DEV_DE_STREETS = ['Mozartstraße', 'Birkenallee', 'Marktgasse']
TRAIN_ES_STREETS = ['Calle de la Paz', 'Calle San Vicente', 'Avenida del Puerto',
 'Paseo de la Alameda', 'Calle Nueva', 'Avenida de la Constitución',
 'Calle del Carmen', 'Ronda de Toledo', 'Calle Ancha', 'Travesía del Mercado',
 'Calle Alfareros', 'Calle Hortelanos', 'Paseo del Río', 'Calle Tejedores',
 'Calle Curtidores', 'Avenida del Mediterráneo']
DEV_ES_STREETS = ['Calle Colón', 'Camino Viejo', 'Avenida de los Naranjos']
TRAIN_IT_STREETS = ['Via Dante', 'Via Mazzini', 'Corso Italia', 'Viale Europa', 'Piazza Cavour',
 'Via Roma', 'Via Leopardi', 'Corso Vittorio Emanuele', 'Via Carducci',
 'Viale dei Mille', 'Via Tiziano', 'Via Petrarca', 'Via Ariosto',
 'Viale Rimembranze', 'Corso Matteotti', 'Via Foscolo']
DEV_IT_STREETS = ['Via Verdi', 'Vicolo Stretto', 'Via Boccaccio']
TRAIN_PT_STREETS = ['Rua do Carmo', 'Rua das Flores', 'Avenida da Liberdade',
 'Travessa do Monte', 'Rua de Santa Catarina', 'Largo do Rossio',
 'Rua da Fonte', 'Rua Nova', 'Rua do Souto', 'Avenida dos Combatentes',
 'Rua do Almada', 'Travessa da Espera']
DEV_PT_STREETS = ['Rua Direita', 'Rua dos Oleiros', 'Rua dos Mercadores']
TRAIN_PL_STREETS = ['ul. Lipowa', 'ul. Długa', 'ul. Ogrodowa', 'ul. Kościelna',
 'al. Niepodległości', 'ul. Polna', 'ul. Leśna', 'ul. Krótka']
DEV_PL_STREETS = ['ul. Słoneczna', 'ul. Szkolna']
TRAIN_NL_STREETS = ['Molenstraat', 'Dorpsweg', 'Kerklaan', 'Prinsengracht', 'Julianalaan',
 'Oosterkade', 'Wilhelminastraat', 'Lange Voorhout']
DEV_NL_STREETS = ['Stationsweg', 'Nieuwstraat']
TRAIN_HOUSE_NAMES = ['The Old Bakery', 'Ivy Lodge', 'The Old Rectory', 'Meadow Cottage',
 'Fern Villa', 'The Old Schoolhouse', 'Beech House', 'Lark Lodge',
 'The Old Smithy', 'Larch Lodge', 'Curlew Cottage', 'The Malthouse',
 'Swallow Barn', 'The Old Dairy', 'Tithe Barn', 'Rookwood House']
DEV_HOUSE_NAMES = ['The Granary', 'Heather Cottage', 'Primrose Villa']
TRAIN_FACILITY_STEMS = ['Stonebridge', 'Wrenbury', 'Elmswood', 'Brookvale', 'Hazelmere',
 'Summerhill', 'Rosebank', 'Queensway', 'Abbeyfield', 'Northcote',
 'Westbourne', 'Highfield', 'Ravensworth', 'Glenview', 'Marlborough',
 'Holmewood', 'Ferndown', 'Lakeside', 'Whitecross', 'Springvale', 'Wolds',
 'Tamarside', 'Severnside']
DEV_FACILITY_STEMS = ['Kingfisher', 'Oakleigh', 'Beaconsfield', 'Cheviot']
TRAIN_SAINTS = ["St Aidan's", "St Brigid's", "St Oswald's", "St Dunstan's"]
DEV_SAINTS = ["St Hilda's"]
TRAIN_COMPANY_STEMS = ['Halcyon', 'Kestrelway', 'Ironwood', 'Sablecourt', 'Tidewell', 'Marlstone',
 'Dunlin', 'Pevensey', 'Arkwright', 'Bellweather', 'Driftwood', 'Emberton',
 'Fallowfield', 'Gantry', 'Harrowgate', 'Lowmoor', 'Merrivale', 'Northstar',
 'Oakenshaw', 'Pendlebury', 'Redgrave', 'Tannerfield', 'Umberleigh',
 'Vantage', 'Wharfedale', 'Zephyr', 'Altamira', 'Bergfeld', 'Costanera',
 'Dornbach', 'Fioravanti', 'Grunwald', 'Hollander', 'Ibérica', 'Jaskółka',
 'Lumière', 'Montagna']
DEV_COMPANY_STEMS = ['Corvane', 'Castlemoor', 'Kelbrook', 'Quillon', 'Yarmouth', 'Estrela',
 'Kastanien']
TRAIN_PRODUCTS = ['Makita DHP 453', 'Stihl MS 181', 'Canon EOS 250', 'Linde H 25',
 'Kärcher K 4', 'Husqvarna 135', 'Hilti TE 30', 'Festool TS 55',
 'Epson ET 2850', 'Lenovo T 14', 'Honda CR-V 1.6', 'Kubota KX 71',
 'Weber Spirit 3', 'Roland FP 30', 'Yamaha P 45', 'Samsung A 54',
 'Philips 5400', 'Wi-Fi 5', 'Ryobi R 18', 'Einhell TE 18', 'Asus B 550',
 'Nikon D 3500', 'Suzuki Swift 1.2', 'Citroën C 3', 'Kyocera M 2040',
 'Beko B 5']
DEV_PRODUCTS = ['DeWalt DCD 796', 'Shimano XT 8100', 'Zanussi Z 716', 'Metabo KGS 216',
 'Hotpoint H 3']
TRAIN_PUBLIC_CITIES = ['Nottingham', 'Cardiff', 'Belfast', 'Plymouth', 'Coventry', 'Stuttgart',
 'Turin', 'Seville', 'Krakow', 'Brno', 'Basel', 'Malmö', 'Aarhus', 'Ghent',
 'Liège', 'Wolverhampton', 'Sunderland', 'Aberystwyth', 'Inverness',
 'Lincoln', 'Dortmund', 'Bremen', 'Eindhoven', 'Tilburg', 'Toulouse']
DEV_PUBLIC_CITIES = ['Swansea', 'Graz', 'Naples', 'Chester']
TRAIN_COUNTRIES = ['NIGERIA', 'KENYA', 'POLAND', 'INDIA', 'JAMAICA', 'ROMANIA', 'PORTUGAL',
 'ZIMBABWE', 'BANGLADESH', 'LITHUANIA', 'PHILIPPINES', 'IRELAND']
DEV_COUNTRIES = ['PAKISTAN', 'SRI LANKA', 'TURKEY']

# Later additions (same freshness check as the pools above).
TRAIN_PRODUCTS += ["Bahco Ergo 23", "Draper Expert 12", "Vango Alpha 300",
                   "Trangia Storm 27", "Thule Epos 2", "Petzl Tikka 3",
                   "Ortlieb Roller 20", "Fiskars Solid 21", "Hozelock Pico 10",
                   "Stanley FatMax 8"]
DEV_PRODUCTS += ["Silverline Pro 16", "Coleman Darwin 4"]
TRAIN_PUBLIC_CITIES += ["Zürich", "Tromsø", "Köln", "Düsseldorf", "Nürnberg",
                        "Göteborg", "Plzeň", "Győr", "Besançon", "Århus"]
DEV_PUBLIC_CITIES += ["Münster", "Łomża"]
TRAIN_PL_STREETS += ["ul. Mickiewicza", "ul. Kwiatowa", "ul. Zielona",
                     "ul. Piłsudskiego", "ul. Sienkiewicza", "al. Jana Pawła II",
                     "ul. Wojska Polskiego", "ul. Grunwaldzka", "ul. Żeromskiego"]
DEV_PL_STREETS += ["ul. Kopernika", "ul. Spacerowa"]
# Streets with no street word: Welsh names, rows of cottages, folds and yards.
TRAIN_PLAIN_STREETS = ["Maes y Coed", "Heol Derwen", "Cae Glas", "Tan y Graig",
                       "Ffordd Las", "The Paddocks", "The Maltings", "Quarry Cottages",
                       "Fold Head", "Low Green", "High Fold", "The Shambles",
                       "Shepherd's Fold", "Llwyn Onn", "Pant Glas", "Rhodfa'r Môr",
                       "Station Cottages", "Canal Side", "Smithy Fold", "Lôn Cariadon",
                       "Trem y Mynydd"]
DEV_PLAIN_STREETS = ["Lime Kilns", "The Wynd", "Colliery Cottages", "Cwrt y Felin"]
# Official records written in English in the Russian order: surname first,
# a patronymic, initials before or after the surname, postcode and city
# ahead of the street. Checked against the gold and the private documents.
TRAIN_RU_LAST = ["Voronin", "Zaitseva", "Tikhonov", "Belyaeva", "Zhukov", "Vinogradova",
                 "Kotov", "Baranova", "Davydova", "Nikitin", "Maksimova", "Sorokin",
                 "Gerasimov", "Ershova", "Kalinin", "Shestakova", "Bogdanov", "Efimova",
                 "Kudryavtsev", "Larionova", "Mironov", "Nazarova", "Osipov"]
DEV_RU_LAST = ["Rodionov", "Savelyeva", "Trofimov", "Ustinova"]
TRAIN_RU_FIRST = ["Anatoly", "Gennady", "Vyacheslav", "Zinaida", "Rodion", "Nadezhda",
                  "Timofey", "Lyubov", "Arkady", "Raisa", "Varvara", "Yaroslav",
                  "Veronika", "Matvey", "Antonina", "Semyon"]
DEV_RU_FIRST = ["Kira", "Gleb", "Alla"]
TRAIN_RU_PATR = ["Borisovich", "Leonidovna", "Anatolyevna", "Grigoryevich",
                 "Stanislavovna", "Fyodorovich", "Arkadyevna", "Konstantinovich",
                 "Valeryevna", "Maksimovich", "Timurovna"]
DEV_RU_PATR = ["Denisovich", "Eduardovna"]
TRAIN_RU_STREETS = ["Sadovaya St.", "Zarechnaya St.", "Parkovaya St.", "Shkolnaya St.",
                    "Polevaya St.", "Rechnaya St.", "Lugovaya St.", "Beregovaya St.",
                    "Solnechnaya St.", "Yubileynaya St.", "Stroiteley Ave.",
                    "Pobedy Prospekt", "Mira Ave.", "Tsentralnaya St.", "Molodyozhnaya St."]
DEV_RU_STREETS = ["Vokzalnaya St.", "Zavodskaya St.", "Pervomayskaya St."]
TRAIN_RU_CITIES = ["Tver", "Pskov", "Vologda", "Ryazan", "Kursk", "Oryol", "Kirov",
                   "Penza", "Tambov", "Ulyanovsk", "Izhevsk", "Kostroma", "Bryansk",
                   "Smolensk", "Yaroslavl"]
DEV_RU_CITIES = ["Lipetsk", "Orenburg", "Tyumen"]
TRAIN_REGIONS = ["Tver Oblast", "Perm Krai", "Republic of Karelia", "Vologda Oblast",
                 "Primorsky Krai", "Lower Saxony", "County Durham", "Province of Utrecht",
                 "Silesian Voivodeship", "Canton of Bern", "Republic of Mordovia",
                 "Kirov Oblast"]
DEV_REGIONS = ["Khabarovsk Krai", "Lesser Poland Voivodeship", "Province of Groningen"]
# Field values that open with a capital and are not names: without them the
# model reads the first word after a label as a person.
LABEL_VALUES = [
    ("Diagnosis on admission", "Chronic sinusitis"), ("Diagnosis", "Viral bronchitis"),
    ("Diagnosis on discharge", "Simple fracture of the wrist"),
    ("Procedure", "Arthroscopic repair"), ("Procedure", "Endoscopic examination"),
    ("Treatment", "Oral antibiotics for seven days"), ("Outcome", "Satisfactory recovery"),
    ("Status", "Pending review"), ("Reason", "Routine inspection"),
    ("Type", "Residential lease"), ("Category", "Light goods"),
    ("Condition", "Good"), ("Payment", "Monthly instalments"),
    ("Grounds", "Unpaid rent"), ("Purpose", "Temporary residence"),
    ("Gender", "Male"), ("Gender", "Female"), ("Marital status", "Single"),
    ("Citizenship", "Not stated"), ("Education", "Secondary vocational"),
    ("Result", "Negative"), ("Recommendation", "Outpatient follow-up"),
    ("Complaint", "Persistent cough"), ("Allergies", "None known"),
    ("Blood group", "Second, positive"), ("Decision", "Approved in full"),
    ("Delivery", "Standard"), ("Priority", "Urgent"), ("Method", "Bank transfer"),
    ("Basis", "Written application"), ("Fitness category", "Limited fitness"),
    ("Remarks", "Original documents returned"),
]
OFFICES = ["Internal Affairs Office", "Migration Office", "Civil Registry Office",
           "Tax Inspectorate", "Pension Fund Office", "Notarial District",
           "District Court", "Housing Department", "Social Protection Office"]
RANKS = ["Col.", "Maj.", "Capt.", "Notary", "Judge", "Inspector", "Registrar", "Chief physician"]
IE_COUNTIES = ["Clare", "Mayo", "Galway", "Donegal", "Wexford", "Meath",
               "Tipperary", "Roscommon", "Leitrim", "Offaly"]

STREET_TYPES = ["Road", "Street", "Lane", "Avenue", "Close", "Drive", "Way",
                "Crescent", "Gardens", "Terrace", "Place", "Row", "Hill",
                "Court", "Grove", "Mews", "Square", "Walk", "Rise", "Parade"]
FACILITY_KINDS = [
    "Surgery", "Medical Practice", "Dental Practice", "Health Centre",
    "Care Home", "Nursing Home", "Day Centre", "Hospice", "Primary School",
    "High School", "Academy", "Sixth Form College", "Nursery",
    "Community Hospital", "Medical Group", "Rehabilitation Centre",
    "Counselling Service", "Clinic", "Hospital", "University",
    "Institute of Technology", "Wellbeing Centre", "Family Practice",
    "Health Trust", "NHS Trust", "Technical Institute", "International School",
    "Family Support Centre", "Speech and Language Centre",
    "Sports Medicine Clinic", "Grammar School", "Polyclinic", "Infirmary",
    "Dialysis Centre", "Therapy Centre", "Counselling Centre",
]
MEDICAL_KINDS = ["Surgery", "Medical Practice", "Dental Practice", "Health Centre",
                 "Community Hospital", "Medical Group", "Rehabilitation Centre",
                 "Clinic", "Hospital", "Family Practice"]
SCHOOL_KINDS = ["Primary School", "High School", "Academy", "Sixth Form College",
                "Nursery"]
CARE_KINDS = ["Care Home", "Nursing Home", "Day Centre", "Hospice",
              "Community Hospital", "Rehabilitation Centre", "Health Centre"]
EDUCATION_KINDS = SCHOOL_KINDS + ["University", "Institute of Technology"]
FACILITY_LEADS = ["Colegio", "Instituto", "Clínica", "Klinik", "Hochschule",
                  "Liceo", "Escuela", "Universidad"]
COMPANY_TAILS = [
    "Plumbing & Heating Ltd", "Motors", "Insurance Services", "Telecom",
    "Removals", "Property Management", "Solicitors LLP", "Accountancy",
    "Travel", "Logistics GmbH", "Assekuranz AG", "S.r.l.", "Lda",
    "Sp. z o.o.", "B.V.", "Leasing Ltd", "Foods", "Engineering",
    "Packaging", "Print & Design", "Garden Machinery", "Car Hire",
    "Building Supplies", "Recruitment", "Couriers", "Software",
]
# A company is as often two capitalised words with no legal form.
COMPANY_LEADS = ["Autohaus", "Autofficina", "Transportes", "Bäckerei", "Carrozzeria",
                 "Tischlerei", "Gestoría", "Drukarnia", "Garage", "Farmacia",
                 "Pasticceria", "Talleres", "Piekarnia", "Spedition"]
COMPANY_NOUNS = ["Spedycja", "Transporte", "Trasporti", "Assekuranz",
                 "Assicurazioni", "Seguros", "Transport", "Bau", "Technik",
                 "Elektro", "Costruzioni", "Mudanzas", "Budownictwo", "Reisen"]
LEGAL_FORMS = ["", "", " GmbH", " S.r.l.", " Sp. z o.o.", " S.L.", " Lda", " Ltd",
               " AG", " B.V.", " S.A."]
PLACE_KINDS = ["Country Park", "Nature Trail", "Leisure Centre", "Retail Park",
               "Railway Station", "Sports Ground", "Village Hall", "Library",
               "Museum", "Garden Centre", "Bus Station", "Wildlife Park",
               "Business Park", "Marina", "Golf Club", "Community Garden"]
NATIONALITIES = ["Kenyan", "Romanian", "Indian", "Irish", "Jamaican",
                 "Lithuanian", "Filipino", "Nigerian", "Portuguese", "Turkish",
                 "Ghanaian", "Bulgarian", "Colombian", "Moroccan", "Latvian"]
ID_DOCUMENTS = ["passport", "identity card", "driving licence", "residence permit"]
PRODUCT_NOUNS = ["hose reel", "secateurs", "drill", "trimmer", "headset", "printer",
                 "stove", "rucksack", "saw", "charger", "lamp", "roof rack"]
# Sentences that open with an ordinary capitalised word: without them a
# sentence-initial capital reads as a name (candidate 3, first training run).
FILLERS = [
    "Parking is restricted to permit holders.", "Deliveries arrive before noon.",
    "Invoices are due within thirty days.", "Keys are kept in the site office.",
    "Visitors must sign in at reception.", "Lunch is provided on both days.",
    "Receipts were filed with the claim.", "Cycling is not permitted on the path.",
    "Recycling goes out on alternate weeks.", "Minutes were approved without changes.",
    "Pets are welcome in the garden only.", "Refunds take five working days.",
    "Uniform must be worn on site.", "Helmets are issued at the gate.",
    "Payment was taken by direct debit.", "Attendance has improved this term.",
    "Certificates were posted last week.", "Drawings are attached to this note.",
    "Meter readings were taken at the start.", "Breakfast is served from seven.",
    "Tickets cannot be exchanged.", "Scaffolding comes down next week.",
    "Feedback is welcome by email.", "Stock was counted twice.",
    "Training continues in the spring.", "Samples were sent to the lab.",
    "Entrance B is closed for repairs.", "Keys A and B are in the safe.",
    "Use option C on the keypad.", "Block D has no lift.",
    "Waste is collected from gate A.", "Zone B floods in heavy rain.",
]
TASKS = ["boiler service at 10", "tyre check before the weekend", "eye test on Friday",
         "booster appointment next week", "form to sign by Thursday",
         "parcel waiting at reception", "meter reading due", "library books overdue",
         "rent review meeting", "dental check-up at 14:30"]
TITLES = ["Mr", "Mrs", "Ms", "Miss", "Mx", "Dr", "Dr.", "Prof.", "Rev.",
          "Sr.", "Sra.", "Herr", "Frau", "Dott.", "Dott.ssa", "Sig.",
          "Sig.ra", "Eng.", "Mevr.", "Pan", "Pani"]
EN_TITLES = ["Mr", "Mrs", "Ms", "Miss", "Dr", "Prof."]
JOBS = ["Warehouse Supervisor", "Payroll Officer", "Site Manager",
        "Lettings Negotiator", "Claims Handler", "Practice Manager",
        "Fleet Coordinator", "Accounts Assistant", "Shift Leader",
        "Senior Buyer", "Planning Engineer", "Customer Adviser",
        "Head of Operations", "Service Technician", "Team Leader"]
RELATIONS = ["MOTHER", "FATHER", "SISTER", "BROTHER", "SPOUSE", "PARTNER",
             "DAUGHTER", "SON", "AUNT", "UNCLE"]
GREETINGS3 = ["Hi", "Hello", "Dear", "Good morning", "Good afternoon", "Hey"]
REMINDERS = ["Reminder for", "Message for", "Note for"]
SIGNOFFS = ["Thanks", "Regards", "Kind regards", "Best wishes", "Cheers",
            "Best", "Many thanks", "Best regards", "Yours"]
ROLE_ADDRESSEES = ["Householder", "Customer", "Finance", "Committee", "Team",
                   "Colleagues", "Members", "Residents", "Supplier",
                   "Accounts", "Sir or Madam", "Tenant", "Parent or Guardian",
                   "Visa Officer", "Hiring Manager", "Subscriber", "Occupier",
                   "Neighbour", "Volunteers", "Policyholder", "Cardholder",
                   "Parents", "Driver", "Keyholder", "Landlord", "Applicant",
                   "Payroll", "Reception", "Homeowner", "Passenger"]
TEAM_SIGNOFFS = ["The Billing Team", "Customer Services", "The Lettings Desk",
                 "Accounts Payable", "The Membership Office", "Site Office",
                 "The Admissions Team", "Facilities Helpdesk"]
SITE_WORDS = ["plant", "depot", "warehouse", "hub", "branch", "office",
              "showroom", "terminal", "yard", "store"]
AWARDS = ["Scholarship", "Award", "Prize", "Bursary", "Fellowship",
          "Travel Grant", "Apprenticeship Medal", "Lecture"]
TOPICS = ["rota for next week", "invoice query", "site visit", "key handover",
          "training day", "parking permits", "stock count", "annual review",
          "minibus booking", "new starter paperwork"]
DOB_LABELS3 = ["Date of birth:", "DOB:", "D.O.B.:", "d.o.b.", "Born:",
               "born on", "born", "birthday:", "Birth date:",
               "Date of birth (DD/MM/YYYY):", "date of birth"]
DATE_LABELS = ["Date:", "Issued:", "Purchased:", "Report date:",
               "Start date:", "Valid until", "Appointment:", "Enrolled:",
               "Signed on", "Received:", "Next review:", "Completed:",
               "Policy start:", "Invoice date:", "Collected on",
               "Normal retirement date:", "Expiry date:", "Due date:",
               "Completion date:", "Date of issue:", "Date of entry:",
               "Renewal date:", "Date joined:", "Leaving date:",
               "Date of last visit:", "Hearing date:"]
MONTHS3 = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
INTL = {
    "es": {"name": "Nombre y apellidos", "dob": "Fecha de nacimiento",
           "addr": "Dirección", "date": "Fecha", "tel": "Teléfono",
           "title": "SOLICITUD DE INSCRIPCIÓN", "school": "Centro",
           "leads": ["Colegio", "Instituto", "Clínica", "Escuela"]},
    "it": {"name": "Nome e cognome", "dob": "Data di nascita",
           "addr": "Indirizzo", "date": "Data", "tel": "Telefono",
           "title": "MODULO DI ISCRIZIONE", "school": "Istituto",
           "leads": ["Liceo", "Instituto", "Clínica"]},
    "pt": {"name": "Nome completo", "dob": "Data de nascimento",
           "addr": "Morada", "date": "Data", "tel": "Telefone",
           "title": "FICHA DE INSCRIÇÃO", "school": "Estabelecimento",
           "leads": ["Colégio", "Instituto", "Clínica"]},
    "de": {"name": "Name", "dob": "Geburtsdatum", "addr": "Anschrift",
           "date": "Datum", "tel": "Telefon", "title": "ANMELDEFORMULAR",
           "school": "Einrichtung", "leads": ["Klinik", "Hochschule"]},
    "pl": {"name": "Imię i nazwisko", "dob": "Data urodzenia",
           "addr": "Adres", "date": "Data", "tel": "Telefon",
           "title": "FORMULARZ ZGŁOSZENIOWY", "school": None, "leads": []},
    "nl": {"name": "Naam", "dob": "Geboortedatum", "addr": "Adres",
           "date": "Datum", "tel": "Telefoon", "title": "AANMELDFORMULIER",
           "school": None, "leads": []},
}
INTL_EN = {"name": "Name", "dob": "Date of birth", "addr": "Address",
           "date": "Date", "tel": "Phone"}


def pools_for(split):
    """The round-3 pools of one split ("TRAIN" or "DEV") as a dict."""
    names = ("FIRST", "LAST", "PARTICLE_LAST", "STREET_STEMS", "UK_TOWNS",
             "DE_TOWNS", "ES_TOWNS", "IT_TOWNS", "PT_TOWNS", "PL_TOWNS",
             "NL_TOWNS", "IE_TOWNS", "DE_STREETS", "ES_STREETS", "IT_STREETS",
             "PT_STREETS", "PL_STREETS", "NL_STREETS", "HOUSE_NAMES",
             "FACILITY_STEMS", "SAINTS", "COMPANY_STEMS", "PRODUCTS",
             "PUBLIC_CITIES", "COUNTRIES", "PLAIN_STREETS", "RU_LAST", "RU_FIRST",
             "RU_PATR", "RU_STREETS", "RU_CITIES", "REGIONS")
    return {name.lower(): globals()[f"{split}_{name}"] for name in names}


def render(marked):
    """Strip the markers; return the text and its spans with offsets."""
    text, spans, pos = [], [], 0
    pattern = re.compile(f"{MARK_OPEN}([A-Z_]+){MARK_SEP}(.*?){MARK_CLOSE}", re.S)
    length = 0
    for match in pattern.finditer(marked):
        before = marked[pos:match.start()]
        text.append(before)
        length += len(before)
        label, value = match.group(1), match.group(2)
        spans.append({"text": value, "label": label,
                      "start": length, "end": length + len(value)})
        text.append(value)
        length += len(value)
        pos = match.end()
    text.append(marked[pos:])
    text = "".join(text)
    assert MARK_OPEN not in text and MARK_CLOSE not in text, "nested marker"
    return text, spans


class Builder3:
    """Random pieces for one round-3 document."""

    def __init__(self, rng, pools):
        self.rng = rng
        self.pools = pools
        self.decoys = []

    # ---- markers
    def _mark(self, value, label):
        return f"{MARK_OPEN}{label}{MARK_SEP}{value}{MARK_CLOSE}"

    def P(self, value):
        return self._mark(value, "PERSON")

    def A(self, value):
        return self._mark(value, "ADDRESS")

    def D(self, value):
        return self._mark(value, "DATE_OF_BIRTH")

    def O(self, value):
        return self._mark(value, "SENSITIVE_ORGANIZATION")

    def decoy(self, value, kind):
        self.decoys.append((value, kind))
        return value

    def pick(self, values):
        return self.rng.choice(values)

    # ---- names (plain strings; the genre decides what to mark)
    def first(self):
        return self.pick(self.pools["first"])

    def surname(self):
        roll = self.rng.random()
        last = self.pools["last"]
        if roll < 0.45:
            return self.pick(last)
        if roll < 0.72:
            left, right = self.rng.sample(last, 2)
            return f"{left}-{right}"
        if roll < 0.92:
            return self.pick(self.pools["particle_last"])
        left, right = self.rng.sample(last, 2)
        return f"{left} {right}"

    def full(self):
        """(first, surname, "First Surname"), sometimes with a middle initial."""
        first, last = self.first(), self.surname()
        if self.rng.random() < 0.08:
            return first, last, f"{first} {self.pick(string.ascii_uppercase)}. {last}"
        return first, last, f"{first} {last}"

    def initialled(self, first=None, last=None):
        first = first or self.first()
        last = last or self.surname()
        return f"{first[0]}. {last}"

    def ru_name(self, style=None):
        """A name in the Russian order; the style picks the layout."""
        last, first, patr = (self.pick(self.pools["ru_last"]), self.pick(self.pools["ru_first"]),
                             self.pick(self.pools["ru_patr"]))
        style = self.rng.randrange(5) if style is None else style
        if style == 0:
            return f"{last} {first} {patr}"
        if style == 1:
            return f"{last} {first[0]}. {patr[0]}."
        if style == 2:
            return f"{first[0]}. {patr[0]}. {last}"
        if style == 3:
            return f"{first} {patr} {last}"
        return f"{first} {last}"

    def ru_city(self):
        return self.decoy(self.pick(self.pools["ru_cities"]), "decoy_city")

    def region(self):
        return self.decoy(self.pick(self.pools["regions"]), "decoy_region")

    def ru_address(self):
        r = self.rng
        tail = self.pick(["", f", apt. {r.randint(1, 180)}", f", building {r.randint(1, 4)}",
                          f", building {r.randint(1, 4)}, apt. {r.randint(1, 180)}"])
        return (f"{r.randint(100000, 690000)}, {self.pick(self.pools['ru_cities'])}, "
                f"{self.pick(self.pools['ru_streets'])}, {r.randint(1, 160)}{tail}")

    def label_value(self):
        label, value = self.pick(LABEL_VALUES)
        return f"{label}: {self.decoy(value, 'decoy_field_value')}"

    def title(self, english=False):
        return self.pick(EN_TITLES if english else TITLES)

    # ---- addresses
    def uk_postcode(self):
        letters = "ABCDEFGHJKLMNPRSTUWYZ"
        area = self.pick(letters) + (self.pick(letters) if self.rng.random() < 0.8 else "")
        return (f"{area}{self.rng.randint(1, 28)} {self.rng.randint(1, 9)}"
                f"{self.pick(letters)}{self.pick(letters)}")

    def uk_street(self):
        if self.rng.random() < 0.15:
            return f"{self.rng.randint(1, 60)} {self.pick(self.pools['plain_streets'])}"
        return (f"{self.rng.randint(1, 180)} {self.pick(self.pools['street_stems'])} "
                f"{self.pick(STREET_TYPES)}")

    def uk_address(self):
        parts = []
        roll = self.rng.random()
        if roll < 0.2:
            parts.append(self.pick(self.pools["house_names"]))
        elif roll < 0.5:
            parts.append(f"{self.pick(['Flat', 'Flat', 'Apartment', 'Unit', 'Room'])} "
                         f"{self.rng.randint(1, 40)}{self.pick(['', '', '', 'A', 'B'])}")
        parts.append(self.uk_street())
        town = self.pick(self.pools["uk_towns"])
        roll = self.rng.random()
        if roll < 0.55:
            parts.append(f"{town} {self.uk_postcode()}")
        elif roll < 0.8:
            parts.extend([town, self.uk_postcode()])
        else:
            parts.append(town)
        return ", ".join(parts)

    def foreign_address(self, country=None):
        country = country or self.pick(["de", "es", "it", "pt", "pl", "nl", "ie"])
        n, r = self.rng.randint(1, 140), self.rng
        if country == "de":
            return (f"{self.pick(self.pools['de_streets'])} {n}, "
                    f"{r.randint(10000, 99999)} {self.pick(self.pools['de_towns'])}")
        if country == "es":
            floor = self.pick(["", "", f", {r.randint(1, 6)}º{self.pick('ABCD')}",
                               f", {r.randint(1, 6)}º izq."])
            return (f"{self.pick(self.pools['es_streets'])} {n}{floor}, "
                    f"{r.randint(10000, 52999)} {self.pick(self.pools['es_towns'])}")
        if country == "it":
            return (f"{self.pick(self.pools['it_streets'])} {n}, "
                    f"{r.randint(10000, 98999)} {self.pick(self.pools['it_towns'])}")
        if country == "pt":
            floor = self.pick(["", "", f", {r.randint(1, 5)}.º Esq", f", {r.randint(1, 5)}.º Dto"])
            return (f"{self.pick(self.pools['pt_streets'])} {n}{floor}, "
                    f"{r.randint(1000, 8999)}-{r.randint(100, 999)} "
                    f"{self.pick(self.pools['pt_towns'])}")
        if country == "pl":
            flat = self.pick(["", f"/{r.randint(1, 40)}", f" m. {r.randint(1, 40)}"])
            return (f"{self.pick(self.pools['pl_streets'])} {n}{flat}, "
                    f"{r.randint(10, 99)}-{r.randint(100, 999)} "
                    f"{self.pick(self.pools['pl_towns'])}")
        if country == "nl":
            return (f"{self.pick(self.pools['nl_streets'])} {n}, {r.randint(1000, 9999)} "
                    f"{self.pick('ABCDEGHJKLNPRSTXZ')}{self.pick('ABCDEGHJKLNPRSTXZ')} "
                    f"{self.pick(self.pools['nl_towns'])}")
        eircode = (f"{self.pick('ACDEFHKNPRTVWXY')}{r.randint(10, 99)} "
                   f"{self.pick('ACDEFHKNPRTVWXY')}{r.randint(0, 9)}"
                   f"{self.pick('ACDEFHKNPRTVWXY')}{r.randint(0, 9)}")
        county = f", Co. {self.pick(IE_COUNTIES)}" if r.random() < 0.5 else ""
        return f"{self.uk_street()}, {self.pick(self.pools['ie_towns'])}{county}, {eircode}"

    def address(self):
        return self.uk_address() if self.rng.random() < 0.6 else self.foreign_address()

    # ---- dates
    def _date_value(self, year, style=None):
        m, d = self.rng.randint(1, 12), self.rng.randint(1, 28)
        month = MONTHS3[m - 1]
        style = self.rng.randrange(8) if style is None else style
        if style == 0:
            return f"{d:02d}/{m:02d}/{year}"
        if style == 1:
            return f"{year}-{m:02d}-{d:02d}"
        if style == 2:
            return f"{d:02d}.{m:02d}.{year}"
        if style == 3:
            return f"{d:02d}-{m:02d}-{year}"
        if style == 4:
            return f"{d} {month[:3]} {year}"
        if style == 5:
            return f"{d} {month} {year}"
        if style == 6:
            suffix = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd"}.get(d, "th")
            return f"{d}{suffix} {month} {year}"
        return f"{month} {d}, {year}"

    def dob_value(self, style=None):
        return self._date_value(self.rng.randint(1942, 2012), style)

    def dob(self, label=None):
        """A birth label followed by a marked birth date."""
        label = self.pick(DOB_LABELS3) if label is None else label
        return f"{label} {self.D(self.dob_value())}"

    def plain_date(self, style=None):
        """An ordinary date in the same formats and years as birth dates."""
        value = self._date_value(self.rng.randint(1988, 2027), style)
        return self.decoy(value, "decoy_date")

    def labelled_date(self):
        return f"{self.pick(DATE_LABELS)} {self.plain_date()}"

    def year(self):
        return self.decoy(str(self.rng.randint(1996, 2026)), "decoy_year")

    def year_range(self):
        start = self.rng.randint(2004, 2021)
        end = self.pick([str(start + self.rng.randint(1, 5)), "present"])
        return self.decoy(f"{start}{self.pick(['–', '-', ' to '])}{end}", "decoy_year_range")

    # ---- organizations and things
    def facility_name(self, kinds=None):
        """A facility; `kinds` narrows the type so the document reads true."""
        roll = self.rng.random()
        stem = self.pick(self.pools["facility_stems"])
        if kinds is None and roll < 0.12:
            return f"{self.pick(FACILITY_LEADS)} {stem}"
        kind = self.pick(kinds or FACILITY_KINDS)
        if roll < 0.27:
            return f"{self.pick(self.pools['saints'])} {kind}"
        if roll < 0.37:  # a facility named after its street
            return (f"{self.pick(self.pools['street_stems'])} "
                    f"{self.pick(['Road', 'Lane', 'Street', 'Park'])} {kind}")
        return f"{stem} {kind}"

    def author_facility(self, kinds=None):
        return self.decoy(self.facility_name(kinds), "decoy_author_org")

    def company(self, kind="decoy_company"):
        stem, roll = self.pick(self.pools["company_stems"]), self.rng.random()
        if roll < 0.6:
            name = f"{stem} {self.pick(COMPANY_TAILS)}"
        elif roll < 0.8:
            name = f"{self.pick(COMPANY_LEADS)} {stem}{self.pick(LEGAL_FORMS)}"
        else:
            name = f"{stem} {self.pick(COMPANY_NOUNS)}{self.pick(LEGAL_FORMS)}"
        return self.decoy(name, kind)

    def place(self):
        """A named public place: not an address and not a facility."""
        name = f"{self.pick(self.pools['facility_stems'])} {self.pick(PLACE_KINDS)}"
        return self.decoy(name, "decoy_place")

    def filler(self):
        return self.decoy(self.pick(FILLERS), "decoy_sentence")

    def bic(self):
        letters = "ABCDEFGHIKLMNOPRSTUVZ"
        code = "".join(self.pick(letters) for _ in range(6)) + self.pick(["2L", "MM", "BB", "PP", "22"])
        return self.decoy(code + self.pick(["XXX", "", "XXX", "A1B"]), "decoy_code")

    def iban(self):
        r = self.rng
        return (f"{self.pick(['GB', 'DE', 'ES', 'IT', 'NL', 'PL', 'IE'])}{r.randint(10, 99)} "
                + " ".join(str(r.randint(1000, 9999)) for _ in range(4)) + f" {r.randint(10, 99)}")

    def intl_phone(self):
        r = self.rng
        return (f"+{self.pick([34, 39, 44, 48, 49, 31, 351, 353])} {r.randint(100, 999)} "
                f"{r.randint(10, 99)} {r.randint(10, 99)} {r.randint(10, 99)}")

    def id_document(self):
        number = (self.pick("ABCDEFGHJKLMNPRSTUVWXYZ") + self.pick("ABCDEFGHJKLMNPRSTUVWXYZ")
                  + str(self.rng.randint(100000, 9999999)))
        return (f"{self.decoy(self.pick(NATIONALITIES), 'decoy_nationality')} "
                f"{self.pick(ID_DOCUMENTS)} {number}")

    def item_line(self, n):
        return (f"Item {n} — Name: {self.product()} {self.pick(PRODUCT_NOUNS)}; "
                f"{self.decoy('Qty ' + str(self.rng.randint(1, 6)), 'decoy_quantity')}; "
                f"€{self.rng.randint(9, 480)}.{self.pick(['00', '50', '95', '80'])}")

    def product(self):
        return self.decoy(self.pick(self.pools["products"]), "decoy_product")

    def city(self):
        return self.decoy(self.pick(self.pools["public_cities"]), "decoy_city")

    def site(self):
        return f"{self.pick(['the', 'the', 'our'])} {self.city()} {self.pick(SITE_WORDS)}"

    def job(self):
        return self.decoy(self.pick(JOBS), "decoy_job_title")

    def phone(self):
        return f"0{self.rng.randint(1200, 1999)} {self.rng.randint(100000, 999999)}"

    def clock(self):
        return f"{self.rng.randint(7, 19):02d}:{self.rng.randint(0, 59):02d}"

    def email_of(self, first, last, domain):
        local = re.sub(r"[^a-z]", "", f"{first[0]}{last}".lower())
        return f"{local}@{domain}.example"


# ------------------------------------------------------------ positive genres

def r3_caps_form(b):
    first, last, _ = b.full()
    kin_first, kin_last, _ = b.full()
    lines = [b.pick(["TENANCY APPLICATION", "PATIENT REGISTRATION", "MEMBERSHIP FORM",
                     "VISA APPLICATION", "ENROLMENT FORM"])
             + b.pick([" — PLEASE COMPLETE IN BLOCK CAPITALS", "", " (BLOCK CAPITALS)"])]
    if b.rng.random() < 0.5:
        lines.append(f"SURNAME: {b.P(last.upper())}")
        lines.append(f"{b.pick(['GIVEN NAMES', 'FIRST NAME', 'FORENAMES'])}: {b.P(first.upper())}")
    else:
        lines.append(f"{b.pick(['FULL NAME', 'NAME OF APPLICANT', 'APPLICANT'])}: "
                     f"{b.P(f'{first} {last}'.upper())}")
    if b.rng.random() < 0.5:
        lines.append(f"{b.pick(['MIDDLE NAME', 'PREVIOUS NAMES', 'TITLE'])}: "
                     f"{b.decoy(b.pick(['NONE', 'N/A', 'NOT APPLICABLE']), 'decoy_form_value')}")
    lines.append(f"DATE OF BIRTH: {b.D(b.dob_value(b.pick([0, 3, 5])).upper())}")
    if b.rng.random() < 0.5:
        lines.append(f"COUNTRY OF BIRTH: {b.decoy(b.pick(b.pools['countries']), 'decoy_country')}")
    lines.append(f"{b.pick(['ADDRESS', 'CURRENT ADDRESS', 'HOME ADDRESS'])}: "
                 f"{b.A(b.uk_address().upper())}")
    lines.append(f"OCCUPATION: {b.job().upper()}")
    if b.rng.random() < 0.6:
        label, kinds = b.pick([("GP SURGERY", MEDICAL_KINDS[:4]), ("SCHOOL ATTENDED", SCHOOL_KINDS),
                               ("REGISTERED WITH", MEDICAL_KINDS)])
        lines.append(f"{label}: {b.O(b.facility_name(kinds).upper())}")
    lines.append(f"{b.pick(['NEXT OF KIN', 'EMERGENCY CONTACT', 'GUARANTOR'])}: "
                 f"{b.P(f'{kin_first} {kin_last}'.upper())} ({b.pick(RELATIONS)})")
    lines.append(f"SIGNED: {b.P(b.initialled(first, last).upper())}   DATE: {b.plain_date(0)}")
    return "\n".join(lines)


def r3_letter(b):
    first, last, full = b.full()
    _, _, writer = b.full()
    title = b.title(english=True)
    reason = b.pick([
        "We have reviewed your account and the direct debit will change next month.",
        "Your application has been received and is with the assessment team.",
        "The engineer's visit is confirmed and no payment is due on the day.",
        "Thank you for returning the signed agreement; your copy is enclosed.",
    ])
    return (f"{b.company('decoy_author_org')}\n{b.plain_date(b.pick([5, 6, 7]))}\n\n"
            f"{b.pick(['For the attention of', 'To:', 'Addressee:'])} {title} {b.P(full)}\n\n"
            f"Dear {title} {b.P(last)},\n\n{reason} "
            f"{b.filler()} If anything is unclear, call {b.phone()} and quote the date of "
            f"this letter.\n\n"
            f"{b.pick(['Yours sincerely', 'Yours faithfully', 'Kind regards'])},\n"
            f"{b.P(writer)}\n{b.job()}")


def r3_email(b):
    to_first, from_first, other = b.first(), b.first(), b.first()
    greeting = b.pick(GREETINGS3)
    if b.rng.random() < 0.25:
        addressee = f"{b.P(to_first)} and {b.P(b.first())}"
    else:
        addressee = b.P(to_first)
    body = b.pick([
        f"The {b.pick(TOPICS)} is sorted. {b.P(other)} has the keys and will open up at "
        f"{b.clock()}. {b.filler()}",
        f"{b.filler()} Could you check the {b.pick(TOPICS)} with {b.P(other)} before Friday?",
        f"We meet at {b.place()} at {b.clock()}. {b.filler()} {b.P(other)} is bringing the forms.",
        f"Quick one about the {b.pick(TOPICS)}: {b.P(other)} is off until {b.plain_date(5)}, "
        f"so please send it to me.",
        f"I have moved the {b.pick(TOPICS)} to {b.site()}. Nothing else changes.",
    ])
    signoff = b.pick(SIGNOFFS)
    close = (f"{signoff}, {b.P(from_first)}" if b.rng.random() < 0.4
             else f"{signoff},\n{b.P(from_first)}")
    return f"Subject: {b.pick(TOPICS)}\n\n{greeting} {addressee},\n\n{body}\n\n{close}"


def r3_prose_return(b):
    first, last, full = b.full()
    title = b.title(english=True)
    again = b.P(first) if b.rng.random() < 0.6 else f"{title} {b.P(last)}"
    return b.pick([
        f"STAFF PROFILE\n{b.P(full)} joined {b.company('decoy_employer')} in {b.year()} as a "
        f"{b.job()}. Since then {again} has trained four apprentices and rewritten the handover "
        f"notes. {b.filler()} Colleagues at {b.site()} say {b.P(first)} is the first person "
        f"they call.",
        f"CASE NOTE\nI met {b.P(full)} at {b.site()} on {b.plain_date()}. {again} explained that "
        f"the delivery had been refused twice. {b.filler()} I agreed to ring {b.P(first)} once "
        f"the depot confirms a new slot.",
        f"NEWSLETTER\nThis month we say goodbye to {b.P(full)}, who retires after "
        f"{b.rng.randint(8, 31)} years. {again} started on the night shift and finished as "
        f"{b.job()}. We wish {b.P(first)} a long and happy retirement.",
    ])


def r3_bare_first(b):
    one, two, three = b.rng.sample(b.pools["first"], 3)
    return b.pick([
        f"HANDOVER\nAsk {b.P(one)} to lock the cage before {b.clock()}. {b.filler()} {b.P(two)} "
        f"confirmed the pallet count and {b.P(three)} will sign the sheet in the morning.",
        f"NOTE\n{b.P(one)} rang about the {b.pick(TOPICS)}. I said {b.P(two)} would call back "
        f"after lunch. {b.filler()} If {b.P(one)} rings again, the file is on my desk.",
        f"NOTE\n{b.filler()} {b.filler()} Pick-up is from {b.site()}; {b.P(one)} has the "
        f"paperwork and {b.P(two)} drives.",
        f"ROTA CHANGE\n{b.P(one)}'s shift on Thursday is covered by {b.P(two)}. "
        f"{b.P(three)} keeps the weekend. Tell {b.P(one)} by {b.clock()} today.",
        f"REMINDER\ncc {b.P(one)}, {b.P(two)}\nThe {b.pick(TOPICS)} moves to {b.site()}. "
        f"{b.P(three)} has the new door code.",
    ])


def r3_chat_export(b):
    a_first, _, a_full = b.full()
    _, _, b_full = b.full()
    day = b.plain_date(0)
    style = b.rng.randrange(3)

    def stamp():
        if style == 0:
            return f"[{day}, {b.clock()}]"
        if style == 1:
            return f"[{b.clock()}]"
        return b.clock()

    where = b.pick([
        "",
        f"{stamp()} {b.P(a_full)}: The parcel ships to {b.A(b.uk_address())} within five "
        f"working days.\n",
        f"{stamp()} {b.P(b_full)}: It is at mine, {b.A(b.uk_street())}. Come round after "
        f"{b.clock()}.\n",
    ])
    return (f"{b.pick(['CHAT EXPORT', 'GROUP CHAT', 'MESSAGES'])}\n"
            f"{stamp()} {b.P(a_full)}: Is the {b.pick(TOPICS)} still on for tomorrow?\n{where}"
            f"{stamp()} {b.P(b_full)}: Yes. Thanks {b.P(a_first)}. I will bring the forms.\n"
            f"{stamp()} {b.P(a_full)}: Great, see you at {b.clock()}.\n"
            f"{stamp()} {b.P(b_full)}: {b.pick(['OK', 'Perfect', 'Will do'])}, "
            f"{b.P(a_first)}.")


def r3_cv(b):
    first, last, full = b.full()
    _, _, referee = b.full()
    head = b.pick(["", "CURRICULUM VITAE\n\n", "CV\n"])
    contact = b.pick([
        f"{b.A(b.address())} | {b.phone()} | {b.email_of(first, last, 'mailbox')}",
        f"Address: {b.A(b.address())}\nPhone: {b.phone()}",
    ])
    lines = [f"{head}{b.P(full)}", contact]
    if b.rng.random() < 0.6:
        lines.append(b.dob(b.pick(["Born:", "Date of birth:", "DOB:", "born"])))
    lines.append("Employment")
    for _ in range(b.rng.randint(1, 3)):
        lines.append(f"{b.year_range()}  {b.job()}, {b.company('decoy_employer')}")
    lines.append("Education")
    lines.append(f"{b.O(b.facility_name(EDUCATION_KINDS))}, {b.pick(['BTEC', 'A levels', 'diploma', 'BSc'])}, "
                 f"{b.year()}")
    lines.append(f"Referee: {b.title(english=True)} {b.P(referee)}, {b.job()} at "
                 f"{b.company('decoy_employer')}")
    return "\n".join(lines)


def r3_titled_roles(b):
    roles = b.rng.sample(["Consultant", "Supervisor", "Tenant", "Policyholder", "Driver",
                          "Cliente / Customer", "Mieter / Tenant", "Paziente / Patient",
                          "Named contact", "Witness", "Guarantor", "Passenger", "Consignee",
                          "Shipper", "Sender", "Payee", "Keyholder", "Claimant", "Proposer",
                          "Lead tenant", "Odbiorca / Consignee", "Destinatario / Recipient"], 3)
    org_role = b.pick(["Registered keeper", "Insurer", "Garage", "Supplier", "Landlord",
                       "Managing agent", "Carrier"])
    lines = [b.pick(["CLAIM RECORD", "BOOKING SHEET", "CASE SUMMARY", "SERVICE RECORD",
                     b.company("decoy_author_org") + " — delivery note DN-"
                     + str(b.rng.randint(100000, 999999)),
                     b.company("decoy_author_org") + " — job sheet"])]
    for role in roles:
        _, _, full = b.full()
        lines.append(f"{role}: {b.title()} {b.P(full)}")
    lines.insert(b.rng.randint(1, 3), f"{org_role}: {b.company()}")
    lines.append(f"{b.labelled_date()}. The file stays open until the invoice from "
                 f"{b.company()} is paid.")
    return "\n".join(lines)


def r3_intl_form(b):
    lang = b.pick(sorted(INTL))
    words = INTL[lang]
    bilingual = b.rng.random() < 0.5

    def label(key):
        return f"{words[key]} / {INTL_EN[key]}" if bilingual else words[key]

    _, _, full = b.full()
    lines = [words["title"],
             f"{label('name')}: {b.title() + ' ' if b.rng.random() < 0.3 else ''}{b.P(full)}",
             f"{label('dob')}: {b.D(b.dob_value(b.pick([0, 2, 3, 1])))}",
             f"{label('addr')}: {b.A(b.foreign_address(lang))}",
             f"{label('tel')}: {b.phone()}"]
    if words["leads"] and b.rng.random() < 0.7:
        stem = b.pick(b.pools["facility_stems"])
        lines.append(f"{words['school']}: {b.O(b.pick(words['leads']) + ' ' + stem)}")
    lines.append(f"{label('date')}: {b.plain_date(b.pick([0, 2, 3]))}")
    return "\n".join(lines)


def r3_medical_letter(b):
    first, last, full = b.full()
    _, gp_last, _ = b.full()
    _, _, writer = b.full()
    return (f"{b.author_facility(MEDICAL_KINDS)}\n{b.plain_date(b.pick([5, 6]))}\n\n"
            f"Re: {b.P(full)}, {b.dob()}, {b.A(b.address())}\n\n"
            f"Dear Dr {b.P(gp_last)},\n\n"
            f"I saw {b.pick([b.P(first), 'Mr ' + b.P(last), 'Ms ' + b.P(last)])} in clinic on "
            f"{b.plain_date()}. The symptoms have settled since the course at "
            f"{b.O(b.facility_name(MEDICAL_KINDS))} and no further imaging is needed. "
            f"I will review in six months.\n\n"
            f"Yours sincerely,\n{b.title(english=True)} {b.P(writer)}\n"
            f"{b.pick(['Consultant', 'Specialist Registrar', 'Clinical Lead'])}")


def r3_warranty(b):
    _, _, full = b.full()
    return (f"{b.company('decoy_author_org')} — "
            f"{b.pick(['warranty registration', 'repair ticket', 'order confirmation'])}\n"
            f"Customer: {b.P(full)}\n"
            f"{b.pick(['Address', 'Delivery address', 'Billing address'])}: {b.A(b.address())}\n"
            f"Product: {b.product()}\n"
            f"Purchased: {b.plain_date()}\n"
            f"{b.decoy('Qty ' + str(b.rng.randint(1, 6)), 'decoy_quantity')}, "
            f"{b.pick(['collected in store', 'delivered by courier', 'fitted on site'])}.\n"
            f"{b.pick(['Engine', 'Motor', 'Spare unit', 'Loan unit'])}: {b.product()}, "
            f"{b.rng.randint(2, 90)} {b.pick(['hp', 'kW', 'kg', 'V'])}, serviced in "
            f"{b.pick(MONTHS3)}.")


def r3_hire(b):
    _, _, hirer = b.full()
    _, _, second = b.full()
    start = b.plain_date(0)
    end = b.plain_date(0)
    return (f"{b.pick(['HIRE RECORD', 'EQUIPMENT HIRE', 'LOAN RECORD'])}\nHirer: {b.P(hirer)}\n"
            f"{b.pick(['Item', 'Equipment', 'Model'])}: {b.product()}\n"
            f"Hire agreement {start} to {end}\n"
            f"Collected from {b.site()} at {b.clock()}.\n"
            f"Driving licence: category B, held since {b.year()}.\n"
            f"Additional driver: {b.title(english=True)} {b.P(second)}\n"
            f"Insurer: {b.company()}")


def r3_award_note(b):
    first, _, full = b.full()
    stem = b.pick(b.pools["company_stems"])
    award = b.decoy(f"{b.rng.randint(2019, 2028)} {stem} {b.pick(AWARDS)}", "decoy_year_title")
    return b.pick([
        f"ANNOUNCEMENT\n{b.P(full)} has been awarded the {award}. {b.P(first)} is based at "
        f"{b.site()} and will give a short talk in Room {b.rng.randint(2, 30)}.",
        f"NOTICE\nApplications for the {award} close on {b.plain_date(5)}. Last year's winner, "
        f"{b.P(full)}, now works from {b.site()}.",
        f"MINUTE\nThe panel met in Bay {b.rng.randint(2, 20)} and chose {b.P(full)} for the "
        f"{award}. Route {b.rng.randint(3, 60)} runs late that evening for guests.",
    ])


def r3_reference_letter(b):
    first, last, full = b.full()
    w_first, w_last, _ = b.full()
    return (f"To whom it may concern,\n\n"
            f"I have known {b.P(full)} since {b.year()}. {b.P(first)} rented "
            f"{b.A(b.address())} from me for {b.rng.randint(2, 9)} years and always paid on "
            f"time. {b.filler()} Before that {b.P(first)} worked at "
            f"{b.company('decoy_employer')} as a {b.job()}.\n\nI am happy to be contacted on {b.phone()}.\n\n"
            f"{b.pick(['Signed', 'Yours faithfully', 'Sincerely'])},\n"
            f"{b.P(b.initialled(w_first, w_last))}")


def r3_witness(b):
    names = [b.full()[2] for _ in range(3)]
    _, officer, _ = b.full()
    if b.rng.random() < 0.5:
        people = (f"Witness 1: {b.P(names[0])}, {b.A(b.address())}, {b.intl_phone()}\n"
                  f"Witness 2: {b.title()} {b.P(names[1])}, {b.A(b.address())}, {b.phone()}\n")
    else:
        people = f"Witnesses: {b.P(names[0])}, {b.P(names[1])} and {b.P(names[2])}.\n"
    return (f"INCIDENT LOG — {b.plain_date()}\n{people}"
            f"Statement taken by {b.pick(['PC', 'Sgt', 'Officer', 'Inspector'])} "
            f"{b.P(officer)} at {b.clock()}.\n"
            f"The van left {b.site()} with the rear doors unlatched. {b.filler()} Nobody was "
            f"hurt and the load was recovered.")


def r3_trip_list(b):
    kids = [b.full() for _ in range(3)]
    extra = b.first()
    _, lead, _ = b.full()
    back = kids[b.rng.randrange(3)][0]
    return (f"{b.author_facility(SCHOOL_KINDS)}\n"
            f"{b.pick(['TRIP LIST', 'Consent form'])} — trip to {b.place()}, {b.plain_date(5)}\n"
            f"Group {b.pick('ABC')}: {b.P(kids[0][2])}, {b.P(kids[1][2])}, {b.P(kids[2][2])} "
            f"and {b.P(extra)}.\n"
            f"Lead: {b.pick(['Mrs', 'Mr', 'Ms', 'Miss'])} {b.P(lead)}.\n"
            f"{b.P(back)} needs a packed lunch without nuts. {b.P(extra)}'s consent form is "
            f"still missing. The coach leaves at {b.clock()}.")


def r3_minutes(b):
    people = [b.full() for _ in range(4)]
    return (f"MINUTES — {b.pick(TOPICS)}\n"
            f"Present: {b.P(b.initialled(people[0][0], people[0][1]))}, {b.P(people[1][2])} "
            f"(chair), {b.P(people[2][2])}.\n"
            f"Apologies: {b.P(people[3][2])}.\n"
            f"1. The quote from {b.company()} was accepted.\n"
            f"2. ACTION: {b.P(people[2][0])} to circulate the plan by {b.plain_date(5)}.\n"
            f"3. {b.P(people[1][0])} thanked the volunteers. {b.filler()} Next meeting at "
            f"{b.pick([b.site(), b.place()])}.")


def r3_delivery(b):
    _, _, full = b.full()
    return (f"DELIVERY INSTRUCTIONS\n"
            f"{b.pick(['Deliver to', 'Recipient', 'Ship to'])}: {b.P(full)}, "
            f"{b.A(b.address())}\n"
            f"If nobody is in, leave the parcel with the neighbour at number "
            f"{b.rng.randint(1, 90)}.\n"
            f"Contact: {b.pick([b.phone(), b.intl_phone()])}. {b.filler()}\n"
            f"Driver: {b.P(b.first())}. Route {b.rng.randint(2, 40)}, out of {b.site()} "
            f"at {b.clock()}.")


def r3_kyc(b):
    first, last, full = b.full()
    holder = full.upper() if b.rng.random() < 0.6 else full
    return (f"{b.company('decoy_author_org')}\nCUSTOMER RECORD\n"
            f"Account holder: {b.pick(['MR ', 'MRS ', 'MS ', 'Dr ', ''])}{b.P(holder)}\n"
            f"{b.dob(b.pick(['Date of birth:', 'DOB:', 'Birth date:']))}\n"
            f"Place of birth: {b.city()}\n"
            f"Identity document: {b.id_document()}\n"
            f"Residential address: {b.A(b.address())}\n"
            f"Account: IBAN {b.iban()} (BIC {b.bic()})\n"
            f"Employer: {b.company('decoy_employer')}\n"
            f"Customer since: {b.plain_date()}\n"
            f"Verified by: {b.P(b.initialled())}")


def r3_transport(b):
    _, _, client = b.full()
    return (f"PATIENT TRANSPORT SHEET — {b.plain_date()}\n"
            f"{b.P(client)} collected from {b.O(b.facility_name(CARE_KINDS))} at {b.clock()}, "
            f"taken to {b.O(b.facility_name(MEDICAL_KINDS))} for {b.clock()}.\n"
            f"Escort: {b.P(b.first())}. Wheelchair on board. Return journey booked through "
            f"{b.company()}.")


def r3_register(b):
    rows = []
    for n in range(1, b.rng.randint(4, 6)):
        first, last, _ = b.full()
        rows.append(f"{n}. {b.P(f'{last}, {first}')}")
    return (f"ATTENDANCE REGISTER — {b.plain_date()}\n" + "\n".join(rows)
            + f"\nTaken by: {b.P(b.full()[2])}")


def r3_thread(b):
    _, _, author = b.full()
    return (f"Subject: RE: {b.pick(TOPICS)}\n\n"
            f"{b.pick(GREETINGS3)} {b.P(b.first())},\n\nThat works for us.\n\n"
            f"{b.pick(SIGNOFFS)},\n{b.P(b.first())}\n\n"
            f"> On {b.plain_date(b.pick([5, 6, 7]))} {b.P(author)} wrote:\n"
            f"> Can we move the {b.pick(TOPICS)} to {b.site()}?")


def r3_tenancy(b):
    _, _, tenant = b.full()
    _, landlord_last, _ = b.full()
    return (f"TENANCY SUMMARY\nTenant: {b.title()} {b.P(tenant)}, {b.dob()}\n"
            f"Property: {b.A(b.address())}\n"
            f"Landlord: {b.company()}\n"
            f"Start date: {b.plain_date()}. Rent is paid to IBAN {b.iban()} (BIC {b.bic()}) "
            f"on the {b.rng.randint(1, 28)}th. {b.filler()}\n"
            f"Previous landlord: {b.pick(['Mr', 'Mrs', 'Ms'])} {b.P(landlord_last)}, "
            f"who confirmed no arrears.")


def r3_message_board(b):
    """Numbered messages: role addressees beside named ones."""
    one, two, three = b.rng.sample(b.pools["first"], 3)
    messages = [
        f"{b.pick(['Dear', 'Hi', 'Hello'])} "
        f"{b.decoy(b.pick(ROLE_ADDRESSEES), 'decoy_role_addressee')}, the water will be off "
        f"on {b.pick(['Tuesday', 'Wednesday', 'Friday'])} from {b.clock()}. "
        f"{b.pick(SIGNOFFS)}, {b.decoy(b.pick(TEAM_SIGNOFFS), 'decoy_team_signoff')}",
        f"{b.pick(GREETINGS3)} {b.P(one)}, your parcel is at {b.site()}. {b.filler()} "
        f"{b.pick(SIGNOFFS)}, {b.P(two)}",
        f"{b.pick(REMINDERS)} {b.P(three)}: {b.pick(TASKS)}"
        f"{b.pick(['.', ' at ' + b.O(b.facility_name(MEDICAL_KINDS)) + '.'])}",
        f"{b.pick(['Dear', 'Hello'])} "
        f"{b.decoy(b.pick(ROLE_ADDRESSEES), 'decoy_role_addressee')}, {b.filler().lower()} "
        f"Reply STOP to opt out.",
    ]
    b.rng.shuffle(messages)
    head = b.pick(["Messages from the residents' app", "Text messages sent today",
                   "Notices", "SMS log"])
    return head + "\n\n" + "\n\n".join(f"{n}) {m}" for n, m in enumerate(messages, 1))


def r3_order(b):
    _, _, full = b.full()
    return (f"{b.company('decoy_author_org')} — order confirmation\n"
            f"Placed on {b.plain_date(5)}.\n\n"
            f"{b.item_line(1)}\n{b.item_line(2)}\n"
            f"Dispatch from {b.site()} within two working days.\n\n"
            f"Deliver to: {b.P(full)}, {b.A(b.address())}\n"
            f"Billing name: {b.P(full)}\n"
            f"Paid by card ending {b.rng.randint(1000, 9999)}.")


def r3_appointment(b):
    _, _, full = b.full()
    return (f"{b.pick(['APPOINTMENT CONFIRMATION', 'BOOKING REFERENCE', 'INTERVIEW NOTICE'])}\n"
            f"Applicant: {b.P(full)}, {b.id_document()}, {b.dob(b.pick(['born', 'born on', 'date of birth']))}.\n"
            f"Appointment: {b.plain_date(5)} at {b.clock()}, counter {b.rng.randint(1, 12)}.\n"
            f"{b.filler()} Bring the confirmation and two photographs. "
            f"Home address on file: {b.A(b.address())}.")


def r3_facility_prose(b):
    first, last, full = b.full()
    return b.pick([
        f"PROGRESS NOTE\n{b.P(full)} attends {b.O(b.facility_name(SCHOOL_KINDS))} and has "
        f"settled well this term. {b.filler()} {b.P(first)} was referred to "
        f"{b.O(b.facility_name(MEDICAL_KINDS))} in {b.pick(MONTHS3)} for a hearing test.",
        f"REVIEW\n{b.P(full)} moved into {b.O(b.facility_name(CARE_KINDS))} in {b.year()}. "
        f"The family visits weekly. {b.P(first)} is under the care of "
        f"{b.O(b.facility_name(MEDICAL_KINDS))}. {b.filler()}",
        f"VOLUNTEER FILE\n{b.P(full)} volunteers at {b.O(b.facility_name())} on Saturdays. "
        f"{b.filler()} Before that {b.P(first)} studied at "
        f"{b.O(b.facility_name(EDUCATION_KINDS))} and worked for {b.company('decoy_employer')}.",
        f"DISCHARGE NOTE\n{b.pick(['Mr', 'Mrs', 'Ms'])} {b.P(last)} was discharged from "
        f"{b.O(b.facility_name(MEDICAL_KINDS))} on {b.plain_date()}. Follow-up is with "
        f"{b.O(b.facility_name(MEDICAL_KINDS))}. {b.filler()} {b.P(full)} can be reached on "
        f"{b.phone()}.",
        f"TRANSFER REQUEST\nPlease move the records of {b.P(full)} from "
        f"{b.O(b.facility_name(SCHOOL_KINDS))} to {b.O(b.facility_name(SCHOOL_KINDS))}. "
        f"{b.O(b.facility_name(MEDICAL_KINDS))} holds the health file. {b.filler()}",
    ])


def r3_official_record(b):
    city = b.pick(b.pools["ru_cities"])
    return (f"{b.pick(['CERTIFICATE', 'EXTRACT', 'POWER OF ATTORNEY', 'NOTIFICATION'])} "
            f"No. {b.rng.randint(10, 9999)}\n"
            f"City of {b.decoy(city, 'decoy_city')}, {b.plain_date(2)}\n\n"
            f"{b.pick(['Surname, first name, patronymic', 'Full name', 'Issued to', 'Payer'])}: "
            f"{b.P(b.ru_name(b.pick([0, 0, 3])))}\n"
            f"{b.dob(b.pick(['Date of birth:', 'born', 'born on']))}\n"
            f"Place of birth: city of {b.ru_city()}, {b.region()}\n"
            f"{b.label_value()}\n"
            f"{b.pick(['Registered at', 'Address:', 'Residing at'])} {b.A(b.ru_address())}\n"
            f"Passport issued by the {b.pick(OFFICES)} of {city} on {b.plain_date(2)}.\n"
            f"{b.label_value()}\n"
            f"{b.pick(RANKS)} {b.P(b.ru_name(b.pick([1, 2, 2])))}\nSeal.")


def r3_statement(b):
    rows = []
    for _ in range(b.rng.randint(2, 3)):
        who = b.pick([f"Transfer to account of {b.P(b.ru_name(b.pick([1, 2, 4])))}",
                      f"Payment to {b.company()}", f"Card purchase, {b.site()}",
                      f"Transfer from {b.P(b.ru_name(b.pick([0, 1, 3])))}"])
        rows.append(f"{b.plain_date(2)}  {who}  "
                    f"{b.pick(['-', '+'])}{b.rng.randint(1, 90)},{b.rng.randint(100, 999)}.00")
    return (f"{b.company('decoy_author_org')}\nACCOUNT STATEMENT\n"
            f"Account holder: {b.P(b.ru_name(b.pick([0, 3, 4])))}\n"
            f"Address: {b.A(b.pick([b.ru_address(), b.address()]))}\n"
            f"{b.label_value()}\n" + "\n".join(rows)
            + f"\nBranch: {b.pick(OFFICES + ['Central Branch', 'City Branch'])} of {b.ru_city()}.")


def r3_clinical_summary(b):
    _, _, full = b.full()
    patient = b.pick([full, b.ru_name(0), b.ru_name(3)])
    doctor = b.pick([b.ru_name(2), b.ru_name(1), b.initialled()])
    return (f"{b.author_facility(MEDICAL_KINDS)}\nDISCHARGE SUMMARY\n"
            f"Patient: {b.P(patient)}\n{b.dob()}\n"
            f"{b.label_value()}\n{b.label_value()}\n{b.label_value()}\n"
            f"Admitted {b.plain_date()}, discharged {b.plain_date()}. {b.filler()}\n"
            f"Follow-up at {b.O(b.facility_name(MEDICAL_KINDS))}.\n"
            f"Attending physician: {b.P(doctor)}")


# ------------------------------------------------------------ negative genres

def r3_role_greeting(b):
    return (f"{b.company('decoy_author_org')}\n\n"
            f"{b.pick(['Dear', 'Hi', 'Hello'])} "
            f"{b.decoy(b.pick(ROLE_ADDRESSEES), 'decoy_role_addressee')},\n\n"
            f"The {b.pick(TOPICS)} takes place on {b.plain_date(5)}. Please reply by "
            f"{b.clock()} the day before.\n\n"
            f"{b.pick(SIGNOFFS)},\n{b.decoy(b.pick(TEAM_SIGNOFFS), 'decoy_team_signoff')}")


def r3_product_sheet(b):
    return (f"PRICE LIST — {b.plain_date()}\n"
            f"{b.product()}: {b.decoy('Qty ' + str(b.rng.randint(1, 9)), 'decoy_quantity')}, "
            f"in stock at {b.site()}\n"
            f"{b.product()}: {b.decoy('Qty ' + str(b.rng.randint(1, 9)), 'decoy_quantity')}, "
            f"back in stock {b.plain_date(5)}\n"
            f"{b.item_line(3)}\n"
            f"Distributor: {b.company()}. {b.filler()} Prices exclude VAT.")


def r3_company_notice(b):
    city = b.city()
    return (f"{b.company('decoy_author_org')}\nNOTICE TO STAFF\n"
            f"From {b.plain_date(5)} the night shift moves to {b.site()}. Agency cover comes "
            f"from {b.company('decoy_employer')}. The audit team is in {city} on "
            f"{b.plain_date(5)} and at {b.site()} the week after. {b.filler()} {b.filler()} "
            f"The summer social is at {b.place()}.")


def r3_event_listing(b):
    stem = b.pick(b.pools["company_stems"])
    return (f"EVENTS\n"
            f"{b.decoy(f'{b.rng.randint(2019, 2028)} {stem} {b.pick(AWARDS)}', 'decoy_year_title')}"
            f" — Room {b.rng.randint(2, 30)}, {b.plain_date(5)}\n"
            f"{b.decoy(f'{b.rng.randint(2019, 2028)} Regional Skills {b.pick(AWARDS)}', 'decoy_year_title')}"
            f" — Bay {b.rng.randint(2, 20)}, {b.plain_date(5)}\n"
            f"Sponsored by {b.company()}. Doors open {b.clock()} at {b.place()} in "
            f"{b.city()}. {b.filler()}")


def r3_blank_form(b):
    lang = b.pick(["es", "it", "pt", "de"])
    words = INTL[lang]
    return (f"{b.pick(['HOW TO COMPLETE THIS FORM', 'FORM GUIDANCE', 'NOTES FOR APPLICANTS'])}\n"
            f"Date of birth (DD/MM/YYYY): leave blank if unknown.\n"
            f"MIDDLE NAME: write {b.decoy(b.pick(['NONE', 'N/A']), 'decoy_form_value')} "
            f"if you have none.\n"
            f"{words['tel']} / Phone: include the area code.\n"
            f"{words['date']} / Date: use the format {b.plain_date(0)}.\n"
            f"Return the form to {b.site()}.")


def r3_field_sheet(b):
    """Label and value lines with nobody in them."""
    lines = [b.label_value() for _ in range(b.rng.randint(4, 6))]
    return (f"{b.pick(['RECORD SUMMARY', 'CASE SHEET', 'INSPECTION FORM', 'TRIAGE NOTE'])}\n"
            f"City of {b.ru_city()}, {b.region()}\n" + "\n".join(lines)
            + f"\nFiled at the {b.pick(OFFICES)} of {b.ru_city()} on {b.plain_date(2)}.")


def r3_facility_bulletin(b):
    return (f"{b.author_facility(MEDICAL_KINDS + SCHOOL_KINDS)}\nNEWS FOR FAMILIES\n"
            f"The car park closes on {b.plain_date(5)} for resurfacing. Flu clinics run on "
            f"Thursdays from {b.clock()}. Thanks to {b.company()} for the new benches.")


ROUND3_POSITIVE = [r3_caps_form, r3_letter, r3_email, r3_prose_return, r3_bare_first,
                   r3_chat_export, r3_cv, r3_titled_roles, r3_intl_form, r3_medical_letter,
                   r3_warranty, r3_hire, r3_award_note, r3_reference_letter,
                   r3_witness, r3_trip_list, r3_minutes, r3_delivery, r3_kyc, r3_transport,
                   r3_register, r3_thread, r3_tenancy, r3_message_board, r3_order,
                   r3_appointment, r3_facility_prose, r3_facility_prose,
                   r3_official_record, r3_official_record, r3_statement,
                   r3_clinical_summary]
ROUND3_NEGATIVE = [r3_role_greeting, r3_product_sheet, r3_company_notice, r3_event_listing,
                   r3_blank_form, r3_facility_bulletin, r3_field_sheet]


def _word_occurrences(text, value):
    pattern = re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)")
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def consistent(text, spans):
    """No overlap, and no labeled name left unlabeled elsewhere in the text."""
    ordered = sorted((s["start"], s["end"]) for s in spans)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        return False
    for span in spans:
        if span["label"] != "PERSON":
            continue
        for start, end in _word_occurrences(text, span["text"]):
            if not any(s <= start and end <= e for s, e in ordered):
                return False
    return True


def build_round3(rng, pools, n_docs, prefix):
    docs, made = [], 0
    while made < n_docs:
        negative = (made % 5 == 4)
        genre = rng.choice(ROUND3_NEGATIVE if negative else ROUND3_POSITIVE)
        builder = Builder3(rng, pools)
        text, spans = render(genre(builder))
        if not consistent(text, spans):
            continue  # reroll: a name collided with another piece of the text
        docs.append({"source_file": f"{prefix}_{made:03d}_{genre.__name__[3:]}.txt",
                     "text": text, "spans": spans,
                     "decoys": [{"text": value, "kind": kind}
                                for value, kind in builder.decoys]})
        made += 1
    return docs
