<style>
  div.md-sidebar.md-sidebar--primary {
    display: none !important;
  }
  .md-grid {
  max-width: 100%; /* or 100%, if you want to stretch to full-width */
}
</style>

# EDA: Análise Exploratória dos Dados

## Carregando as Bibliotecas 📚

``` py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from forex_python.converter import CurrencyRates
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
```

## Lendo os dados 👀



``` py
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

arquivo = 'data/input/Listings.csv'
arq_rev = 'data/input/Reviews.csv'
arquivo_dic = 'data/input/Listings_data_dictionary.csv'
arq_dic = 'data/input/Reviews_data_dictionary.csv' 
```
!!! info 
    Para você replicar estes códigos será necessário baixar as fontes de dados [aqui](https://drive.google.com/drive/folders/1UVEfA673UWxLGnV0Xwg7TdJj-Hc4YH37?usp=sharing) e carregá-las nesta etapa de leitura.

### Dados dos imóveis 🏠

``` py
df_arquivo_dic = pd.read_csv(arquivo_dic)
df_arquivo_dic.head(50)
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Field</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>listing_id</td>
      <td>Listing ID</td>
    </tr>
    <tr>
      <th>1</th>
      <td>name</td>
      <td>Listing Name</td>
    </tr>
    <tr>
      <th>2</th>
      <td>host_id</td>
      <td>Host ID</td>
    </tr>
    <tr>
      <th>3</th>
      <td>host_since</td>
      <td>Date the Host joined Airbnb</td>
    </tr>
    <tr>
      <th>4</th>
      <td>host_location</td>
      <td>Location where the Host is based</td>
    </tr>
    <tr>
      <th>5</th>
      <td>host_response_time</td>
      <td>Estimate of how long the Host takes to respond</td>
    </tr>
    <tr>
      <th>6</th>
      <td>host_response_rate</td>
      <td>Percentage of times the Host responds</td>
    </tr>
    <tr>
      <th>7</th>
      <td>host_acceptance_rate</td>
      <td>Percentage of times the Host accepts a booking...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>host_is_superhost</td>
      <td>Binary field to determine if the Host is a Sup...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>host_total_listings_count</td>
      <td>Total listings the Host has in Airbnb</td>
    </tr>
    <tr>
      <th>10</th>
      <td>host_has_profile_pic</td>
      <td>Binary field to determine if the Host has a pr...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>host_identity_verified</td>
      <td>Binary field to determine if the Host has a ve...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>neighbourhood</td>
      <td>Neighborhood the Listing is in</td>
    </tr>
    <tr>
      <th>13</th>
      <td>district</td>
      <td>District the Listing is in</td>
    </tr>
    <tr>
      <th>14</th>
      <td>city</td>
      <td>City the Listing is in</td>
    </tr>
    <tr>
      <th>15</th>
      <td>latitude</td>
      <td>Listing's latitude</td>
    </tr>
    <tr>
      <th>16</th>
      <td>longitude</td>
      <td>Listing's longitude</td>
    </tr>
    <tr>
      <th>17</th>
      <td>property_type</td>
      <td>Type of property for the Listing</td>
    </tr>
    <tr>
      <th>18</th>
      <td>room_type</td>
      <td>Type of room type in Airbnb for the Listing</td>
    </tr>
    <tr>
      <th>19</th>
      <td>accommodates</td>
      <td>Guests the Listing accomodates</td>
    </tr>
    <tr>
      <th>20</th>
      <td>bedrooms</td>
      <td>Bedrooms in the Listing</td>
    </tr>
    <tr>
      <th>21</th>
      <td>amenities</td>
      <td>Amenities the Listing includes</td>
    </tr>
    <tr>
      <th>22</th>
      <td>price</td>
      <td>Listing price (in each country's currency)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>minimum_nights</td>
      <td>Minimum nights per booking</td>
    </tr>
    <tr>
      <th>24</th>
      <td>maximum_nights</td>
      <td>Maximum nights per booking</td>
    </tr>
    <tr>
      <th>25</th>
      <td>review_scores_rating</td>
      <td>Listing's overall rating (out of 100)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>review_scores_accuracy</td>
      <td>Listing's accuracy score based on what's promo...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>review_scores_cleanliness</td>
      <td>Listing's cleanliness score (out of 10)</td>
    </tr>
    <tr>
      <th>28</th>
      <td>review_scores_checkin</td>
      <td>Listing's check-in experience score (out of 10)</td>
    </tr>
    <tr>
      <th>29</th>
      <td>review_scores_communication</td>
      <td>Listing's communication with the Host score (o...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>review_scores_location</td>
      <td>Listing's location score within the city (out ...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>review_scores_value</td>
      <td>Listing's value score relative to its price (o...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>instant_bookable</td>
      <td>Binary field to determine if the Listing can b...</td>
    </tr>
  </tbody>
</table>
</div>

Se tentarmos ler os arquivos dos imóveis sem configurar um enconding apropriado vamos obter um erro:

!!! failure "Erro"
    'utf-8' codec can't decode byte 0x81 in position 206899: invalid start byte


Usando o encode *'iso-8859-1'* encontra-se trechos, principalmente olhando na varíavel 'name', com caracteres especiais como '', eu busquei encontrar um padrão nos caracteres especiais para preencher os trechos incorretos da forma correta, mas esse padrão alterava na medida que o idioma mudava, felizmente vi uma [postagem](https://www.linkedin.com/posts/lucianovasconcelosf_recebi-uma-mensagem-do-desafio-do-heitor-activity-7118561545864830978-7cwT?utm_source=share&utm_medium=member_desktop) do [Luciano Vasconcelos](https://www.linkedin.com/in/lucianovasconcelosf/) resolvendo este problema com uma abordagem mais eficiente, onde ele corrigia os problemas com o encoding, eliminando o mal pela raiz.

``` py
# Função para corrigir a codificação de uma string
def fix_encoding(problem_string):
    if isinstance(problem_string, str):
        return problem_string.encode('Windows-1252', errors='ignore').decode('utf-8', errors='ignore')
    else:
        return problem_string

# Lendo o arquivo com codificação UTF-8
df = pd.read_csv(arquivo, encoding='utf-8', encoding_errors='ignore')

# Aplicando a correção de codificação na coluna 'Name'
df['name'] = df['name'].apply(fix_encoding)

# Salvando o arquivo corrigido com codificação UTF-8
df.to_csv('data/output/Listings_new.csv', encoding='utf-8', index=False)

df = pd.read_csv('data/output/Listings_new.csv')
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_since</th>
      <th>host_location</th>
      <th>host_response_time</th>
      <th>host_response_rate</th>
      <th>host_acceptance_rate</th>
      <th>host_is_superhost</th>
      <th>host_total_listings_count</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>neighbourhood</th>
      <th>district</th>
      <th>city</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bedrooms</th>
      <th>amenities</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>instant_bookable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>281420</td>
      <td>Beautiful Flat in le Village Montmartre, Paris</td>
      <td>1466919</td>
      <td>2011-12-03</td>
      <td>Paris, Ile-de-France, France</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>1.0</td>
      <td>t</td>
      <td>f</td>
      <td>Buttes-Montmartre</td>
      <td>NaN</td>
      <td>Paris</td>
      <td>48.88668</td>
      <td>2.33343</td>
      <td>Entire apartment</td>
      <td>Entire place</td>
      <td>2</td>
      <td>1.0</td>
      <td>["Heating", "Kitchen", "Washer", "Wifi", "Long...</td>
      <td>53</td>
      <td>2</td>
      <td>1125</td>
      <td>100.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3705183</td>
      <td>39 m² Paris (Sacre Cœur)</td>
      <td>10328771</td>
      <td>2013-11-29</td>
      <td>Paris, Ile-de-France, France</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>1.0</td>
      <td>t</td>
      <td>t</td>
      <td>Buttes-Montmartre</td>
      <td>NaN</td>
      <td>Paris</td>
      <td>48.88617</td>
      <td>2.34515</td>
      <td>Entire apartment</td>
      <td>Entire place</td>
      <td>2</td>
      <td>1.0</td>
      <td>["Shampoo", "Heating", "Kitchen", "Essentials"...</td>
      <td>120</td>
      <td>2</td>
      <td>1125</td>
      <td>100.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4082273</td>
      <td>Lovely apartment with Terrace, 60m2</td>
      <td>19252768</td>
      <td>2014-07-31</td>
      <td>Paris, Ile-de-France, France</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>1.0</td>
      <td>t</td>
      <td>f</td>
      <td>Elysee</td>
      <td>NaN</td>
      <td>Paris</td>
      <td>48.88112</td>
      <td>2.31712</td>
      <td>Entire apartment</td>
      <td>Entire place</td>
      <td>2</td>
      <td>1.0</td>
      <td>["Heating", "TV", "Kitchen", "Washer", "Wifi",...</td>
      <td>89</td>
      <td>2</td>
      <td>1125</td>
      <td>100.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4797344</td>
      <td>Cosy studio (close to Eiffel tower)</td>
      <td>10668311</td>
      <td>2013-12-17</td>
      <td>Paris, Ile-de-France, France</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>1.0</td>
      <td>t</td>
      <td>t</td>
      <td>Vaugirard</td>
      <td>NaN</td>
      <td>Paris</td>
      <td>48.84571</td>
      <td>2.30584</td>
      <td>Entire apartment</td>
      <td>Entire place</td>
      <td>2</td>
      <td>1.0</td>
      <td>["Heating", "TV", "Kitchen", "Wifi", "Long ter...</td>
      <td>58</td>
      <td>2</td>
      <td>1125</td>
      <td>100.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4823489</td>
      <td>Close to Eiffel Tower - Beautiful flat : 2 rooms</td>
      <td>24837558</td>
      <td>2014-12-14</td>
      <td>Paris, Ile-de-France, France</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>1.0</td>
      <td>t</td>
      <td>f</td>
      <td>Passy</td>
      <td>NaN</td>
      <td>Paris</td>
      <td>48.85500</td>
      <td>2.26979</td>
      <td>Entire apartment</td>
      <td>Entire place</td>
      <td>2</td>
      <td>1.0</td>
      <td>["Heating", "TV", "Kitchen", "Essentials", "Ha...</td>
      <td>60</td>
      <td>2</td>
      <td>1125</td>
      <td>100.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>

Como já podemos observar acima, os caracteres especiais estão corretos, para validação, eu filtrei trechos que contém outros idiomas, fui alterando o nome do país e rodando o script novamente, como consulta deixo o último país que verifiquei, a Tailândia, lembrando que o objetivo não era traduzir o alfabeto tailandês para o alfabeto romano, o idioma dos nomes permecerá conforme foram registrados.

``` py
filtered_df = df[df['host_location'].str.contains('Thailand', case=False, na=False)]
print(filtered_df['name'].head(15))
```
```
12016                        Belleville Serviced Apartment
12017                           Studio with Private Garden
12126                  Simply space but extraordinary rest
12128                        1BR INFINITY POOL HIPSTERHOOD
12239                     Modern/Cozy 1BR Apt downtown BKK
12241    1BR Rooftop Garden+Gym+Pool Bangkok, 2 mins to...
12242    356 D CondoSathu49, Bangkok (Rama3-Sathorn-Non...
12433                  cozy & great location 350m.from MRT
12434                                 Metro luxe kasetsart
12574                简风格两居室近MRT/10分钟到达大皇宫/4个人/无边泳池/新房/供接机务
12699     1BR with big balcony by the river, near IconSiam
12759                       New Residence (นิว เรสซิเดนซ์)
12855        Cozy apartment, Saphan Taksin BTS & Boat pier
13065            Sitara Place Serviced Apartment and Hotel
13066                                New Room-5mins to MRT
Name: name, dtype: object
```

``` py
filtered_df = df[df['name'].str.contains('', case=False, na=False)]
print(filtered_df['name'].head(15))
```
Busquei por linhas que continham o caractere '' no meio das palavras e não foi encontrado nenhuma linha.

### Dados das reviews ⭐

``` py
df_arq_dic = pd.read_csv(arq_dic)
df_arq_dic.head()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Field</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>listing_id</td>
      <td>Listing ID</td>
    </tr>
    <tr>
      <th>1</th>
      <td>review_id</td>
      <td>Review ID</td>
    </tr>
    <tr>
      <th>2</th>
      <td>date</td>
      <td>Review date</td>
    </tr>
    <tr>
      <th>3</th>
      <td>reviewer_id</td>
      <td>Reviewer ID</td>
    </tr>
  </tbody>
</table>
</div>

``` py
df_rev = pd.read_csv(arq_rev)
df_rev.head()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>review_id</th>
      <th>date</th>
      <th>reviewer_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11798</td>
      <td>330265172</td>
      <td>2018-09-30</td>
      <td>11863072</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15383</td>
      <td>330103585</td>
      <td>2018-09-30</td>
      <td>39147453</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16455</td>
      <td>329985788</td>
      <td>2018-09-30</td>
      <td>1125378</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17919</td>
      <td>330016899</td>
      <td>2018-09-30</td>
      <td>172717984</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26827</td>
      <td>329995638</td>
      <td>2018-09-30</td>
      <td>17542859</td>
    </tr>
  </tbody>
</table>
</div>

## Análise Exploratória dos Dados 🔎

### Proporção das colunas e linhas 📋

``` py
print('Imóveis 🏠 ',df.shape)
print('Reviews ⭐ ',df_rev.shape)
```
```
Imóveis 🏠  (279712, 33)
Reviews ⭐  (5373143, 4)
```

### Observando o Tipo das Variáveis 🔢

``` py
# Imóveis 🏠 
df.dtypes
```
```
listing_id                       int64
name                            object
host_id                          int64
host_since                      object
host_location                   object
host_response_time              object
host_response_rate             float64
host_acceptance_rate           float64
host_is_superhost               object
host_total_listings_count      float64
host_has_profile_pic            object
host_identity_verified          object
neighbourhood                   object
district                        object
city                            object
latitude                       float64
longitude                      float64
property_type                   object
room_type                       object
accommodates                     int64
bedrooms                       float64
amenities                       object
price                            int64
minimum_nights                   int64
maximum_nights                   int64
review_scores_rating           float64
review_scores_accuracy         float64
review_scores_cleanliness      float64
review_scores_checkin          float64
review_scores_communication    float64
review_scores_location         float64
review_scores_value            float64
instant_bookable                object
dtype: object
```

``` py
# Reviews ⭐
df_rev.dtypes
```
```
listing_id      int64
review_id       int64
date           object
reviewer_id     int64
dtype: object
```

### Verificando se há Valores Nulos ❌

``` py
# Imóveis 🏠 
faltantes = (df.isnull().sum()/len(df['listing_id']))*100
print(faltantes)
```
```
listing_id                      0.000000
name                            0.062564
host_id                         0.000000
host_since                      0.058989
host_location                   0.300309
host_response_time             46.040928
host_response_rate             46.040928
host_acceptance_rate           40.429799
host_is_superhost               0.058989
host_total_listings_count       0.058989
host_has_profile_pic            0.058989
host_identity_verified          0.058989
neighbourhood                   0.000000
district                       86.767818
city                            0.000000
latitude                        0.000000
longitude                       0.000000
property_type                   0.000000
room_type                       0.000000
accommodates                    0.000000
bedrooms                       10.523324
amenities                       0.000000
price                           0.000000
minimum_nights                  0.000000
maximum_nights                  0.000000
review_scores_rating           32.678255
review_scores_accuracy         32.788368
review_scores_cleanliness      32.771208
review_scores_checkin          32.809104
review_scores_communication    32.779073
review_scores_location         32.810534
review_scores_value            32.814109
instant_bookable                0.000000
dtype: float64
```

Lidar com dados faltantes é algo que abre bastante discussões sobre abordagem, um caso óbvio, como a varável *'district'*, que contém 87% dos dados vazios, excluirei a coluna inteira.
Outras colunas, como a *'host_identity_verified'*, possui 0,05% de linhas vazias, onde poderia pensar em manter essas linhas nulas, excluir apenas elas ou preenchê-las com valores, caso fosse apropriado. 

Porém, como o objetivo da minha abordagem é trazer um panorama dos imóveis, essa e outras colunas são desnecessárias, excluí-la-eis.

Como nem tudo é perfeito, temos algumas colunas importantes para nossa análise sobre os imóveis que possuem linhas nulas, que são as reviews, mas um imóvel que tem sua linha de review vazia, não teve review. Para simbolizar a falta de review vou atribuir o número zero para essas colunas. 

``` py
# Reviews que contém valores nulos
rev_columns = [
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value'
]

# Preenchendo as colunas com valores nulos com 0
df[rev_columns] = df[rev_columns].fillna(0)
```

!!! question "Pergunta"
    Como separar quem não teve review com quem teve uma review como de fato sendo nota 0? 

Simples, vou contar o número de reviews que cada imóvel teve na base de reviews, aquela com 5 milhões de linhas, e adicionar esta coluna no dataset dos imóveis.

``` py
# Contar a frequência de cada ID
qty_reviews = df_rev['listing_id'].value_counts().reset_index()

# Renomeando as colunas
qty_reviews.columns = ['listing_id', 'qty_reviews']

qty_reviews.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>qty_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17222007</td>
      <td>891</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8637229</td>
      <td>828</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1249964</td>
      <td>796</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32011332</td>
      <td>762</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2399029</td>
      <td>754</td>
    </tr>
  </tbody>
</table>
</div>

Agora é só adicionar essa coluna no nosso dataset de imóveis, se tivéssemos no **Excel**, era um *PROCV*, no **SQL** um *LEFT JOIN*, mas no **Python** utilizaremos o método *merge*:

``` py
merge_df = df.merge(qty_reviews, on='listing_id', how='left')

# Caso não haja reviews, preencha com 0
merge_df['qty_reviews'] = merge_df['qty_reviews'].fillna(0)
```
Só para não passar batido, vamos verificar se no dataset das reviews possuem valores vazios:

``` py
# Reviews ⭐
faltantes = (df_rev.isnull().sum()/len(df_rev['listing_id']))*100
print(faltantes)
```
```
listing_id     0.0
review_id      0.0
date           0.0
reviewer_id    0.0
dtype: float64
```

### Excluindo Colunas Desnecessárias 🧹

``` py
drop_columns = [
'host_since',                      
'host_location',
'host_response_time', 
'host_response_rate',    
'host_acceptance_rate', 
'host_is_superhost', 
'host_total_listings_count',  
'host_has_profile_pic',
'host_identity_verified',
'district'
]


merge_df.drop(drop_columns,axis = 1, inplace = True)

merge_df.head()
```

### Verificando dados duplicados 🎌

O critério que estou seguindo é de validar se a coluna *'listing_id'* possui dados duplicados, já que ela é a coluna de linha única para o dataset dos imóveis, já no dataset das reviews, esta coluna de *'listing_id'* pode repetir, a que não pode neste caso é a *'review_id'*

``` py
duplicatas = merge_df['listing_id'].duplicated()
contagem_duplicatas = duplicatas.sum()

if contagem_duplicatas > 0:
    print(f"Há {contagem_duplicatas} dados duplicados na coluna 'listing_id'.")
else:
    print("Não existem dados duplicados na coluna 'listing_id'.")
```
!!! Success "Resultado"
    Não existem dados duplicados na coluna 'listing_id'.

``` py
# Reviews ⭐
duplicatas = df_rev['review_id'].duplicated()
contagem_duplicatas = duplicatas.sum()

if contagem_duplicatas > 0:
    print(f"Há {contagem_duplicatas} dados duplicados na coluna 'review_id'.")
else:
    print("Não existem dados duplicados na coluna 'review_id'.")
```
!!! warning "Atenção"
    Há 160 dados duplicados na coluna 'review_id'.

``` py
#Excluindo as linhas duplicadas
df_rev.drop_duplicates(subset=['review_id'], inplace=True)
```
   
### Ajustando Valores dos Dados 📏

Embora todas as notas das reviews estão na mesma escala, de 0 a 10 e uma nota geral de 0 a 100, vou mudar a escala delas para ser de 0 a 5, porque esta é a escala que a Airbnb divulga as notas dos imóveis em seu [site](https://www.airbnb.com.br/).

``` py
# Coluna de 0 a 100
merge_df['review_scores_rating'] = (merge_df['review_scores_rating'] / 20).round(2)

# Colunas de 0 a 10

rev_columns = [
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value'
]

merge_df[rev_columns] = (merge_df[rev_columns] / 2).round(2)
```
Outro valor que podemos ver de ajustar é a variável *'price'*, segundo a descrição em seu dicionário ela é:

*'Listing price (in each country's currency)'*

Ou seja, são valores com a moeda local de cada país, se quisermos avaliar os valores, não podemos comparar banana 🍌 com maçã 🍎...

``` py
merge_df['price'].describe()
```
```
count    279712.000000
mean        608.792737
std        3441.826611
min           0.000000
25%          75.000000
50%         150.000000
75%         474.000000
max      625216.000000
Name: price, dtype: float64
```

Ignorando quem anunciou seu imóvel de graça rsrs, temos um valor máximo de 625 mil (dólares, reais, kwanzas?), esses valores são das diárias, então a menos que seja um quarto na ISS, esse valor não é em dólar, mas para solucionar esse problema, podemos listar quais cidades possuímos imóveis anunciados e consequentemente seu país e moeda de origem.

``` py
merge_df['city'].unique()[:100]
```
array(['Paris', 'New York', 'Bangkok', 'Rio de Janeiro', 'Sydney',
       'Istanbul', 'Rome', 'Hong Kong', 'Mexico City', 'Cape Town'],
      dtype=object)

Felizmente temos apenas 10 cidades, então vou aproveitar a viagem e mapear manualmente os países correspondentes, porque essa coluna é útil para um filtro no dashboard.

``` py
# Dicionário que mapeia cada cidade ao seu país correspondente
city_to_country = {
    'Paris': 'France',
    'New York': 'United States',
    'Sydney': 'Australia',
    'Rome': 'Italy',
    'Rio de Janeiro': 'Brazil',
    'Istanbul': 'Turkey',
    'Mexico City': 'Mexico',
    'Bangkok': 'Thailand',
    'Cape Town': 'South Africa',
    'Hong Kong': 'Hong Kong' # Embora Hong Kong seja uma Região Administrativa Especial da China, defini como um país 💪🏼
}

# Função para obter o país com base na cidade
def get_country(row):
    return city_to_country.get(row['city'], 'Unknown')

# Criar uma nova coluna 'country' com base na cidade
merge_df['country'] = merge_df.apply(get_country, axis=1)

# novo DataFrame separado para as taxas de câmbio
exchange_rates = pd.DataFrame({
    'country': ['France', 'United States', 'Australia', 'Italy', 'Brazil', 'Turkey', 'Mexico', 'Thailand', 'South Africa', 'Hong Kong'],
})

c = CurrencyRates()

# A taxa de câmbio de EUR para USD está dando um resultado totalmente errado, porém, USD para EUR está correto, então tive que fazer essa correção
euro_to_usd_rate = 1+(1-c.get_rate('USD', 'EUR'))

def get_exchange_rate(row):
    currency = country_to_currency.get(row['country'])
    if currency == 'EUR':
        return euro_to_usd_rate
    return c.get_rate(currency, 'USD')

# Dicionário que mapeia cada país à sua moeda correspondente
country_to_currency = {
    'France': 'EUR',
    'United States': 'USD',
    'Australia': 'AUD',
    'Italy': 'EUR',
    'Brazil': 'BRL',
    'Turkey': 'TRY',
    'Mexico': 'MXN',
    'Thailand': 'THB',
    'South Africa': 'ZAR',
    'Hong Kong': 'HKD',
}

# Nova coluna 'exchange_rate' no DataFrame exchange_rates
exchange_rates['exchange_rate'] = exchange_rates.apply(get_exchange_rate, axis=1)

# Merge entre o DataFrame merge_df e o DataFrame exchange_rates usando a coluna 'country' como chave
out_df = merge_df.merge(exchange_rates, on='country', how='left')

# Criar uma nova coluna 'price_usd' multiplicando 'price' por 'exchange_rate'
out_df['price_usd'] = (out_df['price'] * out_df['exchange_rate']).round(2)

# O DataFrame out_df agora conterá as colunas 'exchange_rate' e 'price_usd'
out_df.head()
```
O nosso arquivo tratado está praticamente pronto, um último ajuste, apenas para uma questão de filtro no dashboard também, vou trocar ou valores da variável *'instante_bookable'* de 't' e 'f' por um texto:

``` py
# Substitua 't' por 'Instant Bookable' e 'f' por 'Schedule Reservation' na coluna 'instant_bookable'
out_df['instant_bookable'] = out_df['instant_bookable'].replace({'t': 'Instant Bookable', 'f': 'Schedule Reservation'})
```

### Análises Gráficas 📈

#### Gráfico de Distribuição 📊

``` py
# Calcular a média de 'price_usd' por país
average_prices = out_df.groupby('country')['price_usd'].mean().reset_index()

# Ordenar o DataFrame pela média de 'price_usd' em ordem decrescente
average_prices = average_prices.sort_values(by='price_usd', ascending=False)

# Criar um gráfico de barras
plt.figure(figsize=(12, 6))
plt.bar(average_prices['country'], average_prices['price_usd'], color='#ff5a60')
plt.xlabel('País')
plt.ylabel('Média de Preço (USD)')
plt.title('Média de Preço (USD) por País')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Exibir o gráfico
plt.show()
```
![Alt text](img\distribuicao.png)
Este é um gráfico simples, podemos trazê-lo direto no dashboard e se necessário, incrementá-lo lá, então deixaremos para trazer outros gráficos assim para a parte de criação do dashboard.

#### Gráfico de Dispersão 👨🏽‍👨🏽‍👦🏽‍👧🏽

``` py
# Criar um gráfico de dispersão
plt.figure(figsize=(8, 6))
plt.scatter(out_df['price_usd'], out_df['review_scores_rating'], color='#ff5a60', alpha=0.5)
plt.title('Correlação entre Valor do Imóvel e Nota de Avaliação')
plt.xlabel('Valor do Imóvel')
plt.ylabel('Nota da Avaliação')
plt.grid(True)

# Exibir o gráfico
plt.show()
```
![Alt text](img\dispersao.png)
Usando duas dimensões podemos facilmente ver uma concentração na parte superior esquerda, que indica ótimas avaliações e preço menos, podemos observar um acúmulo de imóveis com nota zero, que entra na questão de imóveis sem avaliação que discutimos anteriormente. Além disso, temos um ponto isolado que custa mais de 120 mil dólares, este ponto é:
``` py
selected_columns = ['name', 'neighbourhood', 'city', 'country', 'property_type', 'qty_reviews','price', 'price_usd']
out_df2 = out_df[selected_columns]

out_df2 = out_df2.sort_values(by='price_usd', ascending=False)
out_df2.head(30)
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>neighbourhood</th>
      <th>city</th>
      <th>country</th>
      <th>property_type</th>
      <th>qty_reviews</th>
      <th>price</th>
      <th>price_usd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>181027</th>
      <td>Temporary rentals for Brazilian Cup.</td>
      <td>Sao Cristovao</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Shared room in house</td>
      <td>0.0</td>
      <td>625216</td>
      <td>127783.41</td>
    </tr>
    <tr>
      <th>162251</th>
      <td>Hotel Boutique Maison Salamanca</td>
      <td>Cuauhtemoc</td>
      <td>Mexico City</td>
      <td>Mexico</td>
      <td>Room in boutique hotel</td>
      <td>0.0</td>
      <td>499000</td>
      <td>29095.47</td>
    </tr>
    <tr>
      <th>134601</th>
      <td>B&amp;B Linda House - Double bedroom between Copac...</td>
      <td>Copacabana</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Private room in bed and breakfast</td>
      <td>0.0</td>
      <td>134562</td>
      <td>27502.16</td>
    </tr>
    <tr>
      <th>44515</th>
      <td>3 quartos Elegante na Av Atlantica com Vista P...</td>
      <td>Copacabana</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>0.0</td>
      <td>129233</td>
      <td>26413.01</td>
    </tr>
    <tr>
      <th>245011</th>
      <td>Owesome flat in Botafogo</td>
      <td>Botafogo</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>0.0</td>
      <td>129080</td>
      <td>26381.74</td>
    </tr>
    <tr>
      <th>245787</th>
      <td>Rio Spot Homes T027</td>
      <td>Barra da Tijuca</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>4.0</td>
      <td>129080</td>
      <td>26381.74</td>
    </tr>
    <tr>
      <th>64485</th>
      <td>Apartamento Alma Carioca | RIO136</td>
      <td>Ipanema</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>3.0</td>
      <td>129080</td>
      <td>26381.74</td>
    </tr>
    <tr>
      <th>219194</th>
      <td>Amplo apartamento para famílias</td>
      <td>Lagoa</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>0.0</td>
      <td>129080</td>
      <td>26381.74</td>
    </tr>
    <tr>
      <th>235968</th>
      <td>Studio in Botafogo</td>
      <td>Botafogo</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>97.0</td>
      <td>129080</td>
      <td>26381.74</td>
    </tr>
    <tr>
      <th>239720</th>
      <td>Linda vista com jardim em Santa Teresa</td>
      <td>Santa Teresa</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>33.0</td>
      <td>129080</td>
      <td>26381.74</td>
    </tr>
    <tr>
      <th>137832</th>
      <td>B&amp;B Zul e Verde - Large quadruple room in the ...</td>
      <td>Copacabana</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Private room in bed and breakfast</td>
      <td>0.0</td>
      <td>126233</td>
      <td>25799.86</td>
    </tr>
    <tr>
      <th>134600</th>
      <td>B&amp;B Peixoto - Double ensuite room in Copacabana</td>
      <td>Copacabana</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Private room in bed and breakfast</td>
      <td>0.0</td>
      <td>126233</td>
      <td>25799.86</td>
    </tr>
    <tr>
      <th>134602</th>
      <td>B&amp;B Angela Copa - Double bedroom in Copacabana</td>
      <td>Copacabana</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Private room in bed and breakfast</td>
      <td>0.0</td>
      <td>126233</td>
      <td>25799.86</td>
    </tr>
    <tr>
      <th>134599</th>
      <td>B&amp;B Zul e Verde - Large bedroom in the heart o...</td>
      <td>Copacabana</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Private room in bed and breakfast</td>
      <td>0.0</td>
      <td>126233</td>
      <td>25799.86</td>
    </tr>
    <tr>
      <th>61242</th>
      <td>Apt Claudia Copa</td>
      <td>Copacabana</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>0.0</td>
      <td>126233</td>
      <td>25799.86</td>
    </tr>
    <tr>
      <th>137831</th>
      <td>B&amp;B Zul e Verde - Large triple room in the hea...</td>
      <td>Copacabana</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Private room in bed and breakfast</td>
      <td>0.0</td>
      <td>126233</td>
      <td>25799.86</td>
    </tr>
    <tr>
      <th>244010</th>
      <td>Holiday Rentals for up to 5 people in Copacaba...</td>
      <td>Copacabana</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>1.0</td>
      <td>112205</td>
      <td>22932.78</td>
    </tr>
    <tr>
      <th>211416</th>
      <td>STIO CANTINHO DAS GARÇAS - GUARATIBA #505</td>
      <td>Guaratiba</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire house</td>
      <td>0.0</td>
      <td>111411</td>
      <td>22770.50</td>
    </tr>
    <tr>
      <th>211419</th>
      <td>Casa no Jardim Botanico #512</td>
      <td>Alto da Boa Vista</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire house</td>
      <td>0.0</td>
      <td>110918</td>
      <td>22669.73</td>
    </tr>
    <tr>
      <th>37652</th>
      <td>Esplendida Cobertura com Vista para o Mar</td>
      <td>Barra da Tijuca</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>1.0</td>
      <td>109281</td>
      <td>22335.16</td>
    </tr>
    <tr>
      <th>98808</th>
      <td>Habitacion a 40min del centro historico</td>
      <td>Iztapalapa</td>
      <td>Mexico City</td>
      <td>Mexico</td>
      <td>Private room in house</td>
      <td>0.0</td>
      <td>350000</td>
      <td>20407.64</td>
    </tr>
    <tr>
      <th>111113</th>
      <td>******Paddington Sydney*****</td>
      <td>Woollahra</td>
      <td>Sydney</td>
      <td>Australia</td>
      <td>Private room in house</td>
      <td>2.0</td>
      <td>28613</td>
      <td>18703.58</td>
    </tr>
    <tr>
      <th>210814</th>
      <td>M/Y ONEWORLD - All-Inclusive Superyacht Charter</td>
      <td>Sydney</td>
      <td>Sydney</td>
      <td>Australia</td>
      <td>Boat</td>
      <td>0.0</td>
      <td>23571</td>
      <td>15407.76</td>
    </tr>
    <tr>
      <th>36228</th>
      <td>Cute 1 bedroom flat in the center of Paris.</td>
      <td>Gobelins</td>
      <td>Paris</td>
      <td>France</td>
      <td>Entire apartment</td>
      <td>2.0</td>
      <td>12000</td>
      <td>13059.08</td>
    </tr>
    <tr>
      <th>60735</th>
      <td>Amazing apartment 10P-St Marcel/Mouffetard MASQ</td>
      <td>Gobelins</td>
      <td>Paris</td>
      <td>France</td>
      <td>Entire apartment</td>
      <td>4.0</td>
      <td>11599</td>
      <td>12622.69</td>
    </tr>
    <tr>
      <th>196424</th>
      <td>Room in the most exclusive area in Mexico City</td>
      <td>Miguel Hidalgo</td>
      <td>Mexico City</td>
      <td>Mexico</td>
      <td>Private room in apartment</td>
      <td>3.0</td>
      <td>206499</td>
      <td>12040.45</td>
    </tr>
    <tr>
      <th>202783</th>
      <td>Entire Palace - 8 Apartments at Piazza Di Spagna</td>
      <td>I Centro Storico</td>
      <td>Rome</td>
      <td>Italy</td>
      <td>Entire condominium</td>
      <td>0.0</td>
      <td>10571</td>
      <td>11503.96</td>
    </tr>
    <tr>
      <th>204377</th>
      <td>Ocean View Villa with Pool in Joa - ilive010</td>
      <td>Sao Conrado</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire villa</td>
      <td>0.0</td>
      <td>55978</td>
      <td>11440.94</td>
    </tr>
    <tr>
      <th>222758</th>
      <td>COZY AND CONFORTABLE 2BDR - LB1-0022</td>
      <td>Leblon</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>0.0</td>
      <td>55489</td>
      <td>11341.00</td>
    </tr>
    <tr>
      <th>44312</th>
      <td>COPACABANA RIO - Beautiful Beachfront 4 -BDR #...</td>
      <td>Copacabana</td>
      <td>Rio de Janeiro</td>
      <td>Brazil</td>
      <td>Entire apartment</td>
      <td>0.0</td>
      <td>55311</td>
      <td>11304.62</td>
    </tr>
  </tbody>
</table>
</div>

*'Temporary rentals for Brazilian Cup'* no Rio de Janeiro, acho improvável um imóvel ter este valor de diária mesmo em uma data especial como a de uma Copa do Mundo, além disso, foi por causa deste imóvel que a média dos imóveis do Brasil foi a mais alta entre todos os países, sendo uma surpresa para 0 pessoas que ele não foi alugado nenhuma vez, até porquê, é 120 mil em um quarto compartilhado kkrying 🤣

Além do fato dele ser um imóvel anunciado como um aluguel temporário por causa da Copa do Mundo e ela já passou... Vou remover este dado do dataset.

``` py
out_df = out_df.drop(181027)
```
Desconsiderando este imóvel que acabamos de excluir, os outros imóveis também possuem um valor bem alto, um gráfico que é bom para identificação desses possíveis outliers é o boxplot:

``` py
plt.figure(figsize=(10, 6))
sns.boxplot(data=out_df, x='country', y='price_usd')
plt.title('Distribuição de Preços por País')
plt.xticks(rotation=45)
plt.show()
```
![Alt text](img/boxplot.png)

Entramos no [dilema dos outliers](https://chat.openai.com/share/9f8a8820-1028-4d27-a055-3200ca0e71d2). 

!!! question "Pergunta"
    Poderíamos simplesmente remover esses pontos isolados para ter valores, como médias, mais coerentes com nossos dados reais?

Eu não sou especialista neste assunto de imóveis, então o que me restou foi realizar um cornojob, onde pesquisei no próprio site da Airbnb imóveis nestes locais com filtros para obter os maiores valores, inclusive, faltou um 'Filtrar por maior preço' no site, em alguns casos eu tive que editar o valor direto na URL para filtrar.

- [Rio de Janeiro](https://www.airbnb.com/s/Rio-de-Janeiro--Rio-de-Janeiro--Brazil/homes?adults=2&tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&flexible_trip_lengths%5B%5D=one_week&monthly_start_date=2023-11-01&monthly_length=3&price_filter_input_type=2&price_filter_num_nights=5&channel=EXPLORE&search_type=autocomplete_click&date_picker_type=calendar&source=structured_search_input_header&price_min=18000&query=Rio%20de%20Janeiro%2C%20RJ&place_id=ChIJW6AIkVXemwARTtIvZ2xC3FA)
- [Austrália](https://www.airbnb.com/s/Paddington-Sydney/homes?adults=2&tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&flexible_trip_lengths%5B%5D=one_week&monthly_start_date=2023-11-01&monthly_length=3&price_filter_input_type=2&price_filter_num_nights=5&channel=EXPLORE&search_type=filter_change&date_picker_type=calendar&source=structured_search_input_header&query=Paddington%20NSW%2C%20Australia&place_id=ChIJt4nzrAiuEmsRcMUyFmh9AQU&price_min=2478&tier_ids%5B%5D=2)
- [Cidade do México](https://www.airbnb.com/s/Mexico-City--Mexico-City--Mexico/homes?adults=2&tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&flexible_trip_lengths%5B%5D=one_week&monthly_start_date=2023-11-01&monthly_length=3&price_filter_input_type=2&price_filter_num_nights=5&channel=EXPLORE&search_type=filter_change&date_picker_type=calendar&source=structured_search_input_header&price_min=18000&query=Mexico%20City%2C%20Mexico%20City%2C%20Mexico&place_id=ChIJB3UJ2yYAzoURQeheJnYQBlQ&federated_search_session_id=5cc6ea3e-4781-4ae9-b33f-a6ccf08c2690&pagination_search=true&cursor=eyJzZWN0aW9uX29mZnNldCI6MCwiaXRlbXNfb2Zmc2V0IjowLCJ2ZXJzaW9uIjoxfQ%3D%3D)

Em todos os casos existe imóveis de alto padrão correspondentes aos preços altos encontrados no dataset, então aqueles valores que por mais que sejam muito altos, são reais.

!!! sucess "Decisão:"
    Vou manter todos esses dados, posso na etapa do dashboard criar uma label para filtrar e excluir esses valores da visualização dos dados, para quando for pertinenete.

!!! quote ""
    'Ah! mas você já removeu aquela linha que tinha um quarto compartilhado de 120 mil dólares no RJ'

Bem, já havia dito o quão rídiculo era aquele caso, pelo próprio dataset era possível saber que não se tratava de um imóvel luxuoso, mas como desencargo de consciência, eu também realizei uma pesquisa com imóveis no RJ por mais de 100k dólares e só achei outros belos e majestosos imóveis como o que excluí, o melhor que desta vez tem fotos, vale a pena conferir:

[Rio de Janeiro +100k USD](https://www.airbnb.com/s/Rio-de-Janeiro--Rio-de-Janeiro--Brazil/homes?adults=2&tab_id=home_tab&refinement_paths[]=%2Fhomes&flexible_trip_lengths[]=one_week&monthly_start_date=2023-11-01&monthly_length=3&price_filter_input_type=2&price_filter_num_nights=5&channel=EXPLORE&search_type=autocomplete_click&date_picker_type=calendar&source=structured_search_input_header&price_min=100000&query=Rio%20de%20Janeiro%2C%20RJ&place_id=ChIJW6AIkVXemwARTtIvZ2xC3FA)

Vamos voltar ao nosso gráfico de dispersão, agora sem aquele outlier, os pontos que estavam achatados na esquerda serão melhores distribuídos.
Também vamos aproveitar e incrementar o gráfico, adicionando mais um eixo nele, plotando assim um gráfico em 3D, com:

* Valor do imóvel
* Review do imóvel
* Quantidade de review 


``` py
# Criar um gráfico tridimensional
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = out_df['price_usd']
y = out_df['review_scores_rating']
z = out_df['qty_reviews']

# Scatter plot tridimensional
ax.scatter(x, y, z, c='#ff5a60', marker='o', alpha=0.5)

ax.set_xlabel('price_usd')
ax.set_ylabel('review_scores_rating')
ax.set_zlabel('Popularidade (qrt_reviews)')

plt.title('Relação entre Valor do Imóvel, Nota de Avaliação e Popularidade')

# Exibir o gráfico tridimensional
plt.show()
```
![Alt text](img/distribuicao2.png)
Com este gráfico atual a gente separou o joio do trigo 🌾.
Antes um imóvel com alto valor poderia receber 1 ou 2 feedbacks e ficar bem posicionado, usando a quantidade de reviews sabemos quais são de fato os mais populares e lucrativos.
Se você tem dificuldade em enxergar em um mapa de 3 dimensões, neste gráfico a melhor posição é para frente, onde as revies tendem a 5, para cima, onde a popularidade (etiqueta cortada), está tendendo a 800 e por fim, onde o preço está para direta, chegando até 30 mil.

Conseguiu entender? Bom! Então vamos complicar mais um pouco... Podemos agrupor por instant_bookable, essa variável basicamente determina o que pode ser alugado de imediato e o que não pode.
Casas sem hospedeiros podem ser alugadas instantaneamente, já as casas que estão alugadas, precisam ser reservadas para o final do contrato atual.

Então mesmo ocupando uma ótima posição no gráfico, se a casa está disponível, é sinal que dinheiro por ela não tá entrando no caixa.

``` py
color_map = {'Instant Bookable': '#ff5a60', 'Schedule Reservation': 'black'}
colors = out_df['instant_bookable'].map(color_map)

# Criar um gráfico tridimensional
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = out_df['price_usd']
y = out_df['review_scores_rating']
z = out_df['qty_reviews']

# Scatter plot tridimensional
ax.scatter(x, y, z, c=colors, marker='o', alpha=0.5)

ax.set_xlabel('price_usd')
ax.set_ylabel('review_scores_rating')
ax.set_zlabel('Popularidade (qty_reviews)')

# Adicionar legenda
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Instant Bookable', markerfacecolor='#ff5a60', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Schedule Reservation', markerfacecolor='black', markersize=10)]
ax.legend(handles=legend_elements)



plt.title('Relação entre Valor do Imóvel, Nota de Avaliação e Popularidade (agrupado por instant_bookable)')

# Exibir o gráfico tridimensional
plt.show()
```
![Alt text](img/distribuicao3.png)

#### Mapa de Calor 🔥🗺️


``` py
# Selecionar as colunas numéricas
numeric_columns = out_df.select_dtypes(include=['number'])

# Calcular a matriz de correlação
correlation_matrix = numeric_columns.corr()

# Criar um mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt='.2f')
plt.title('Mapa de Calor das Variáveis')
plt.show()
```
![Alt text](img/mapa.png)
Tirando as correlações óbvias, como notas das reviews que são correlacionadas, assim como o valor da moeda e seu respectivo valor convertido em dólar e afins, não achei nenhum ponto muito interessante, mas este heatmap é apenas das variáveis numéricas, vamos finalizar a exploração em Python e continuar em nosso dashboard.



### Definindo um Score 🥇

Conforme vimos no [gráfico de dispersão](#2), quando trazemos uma contextualização da situação de negócio para os dados analisados, observar uma váriavel isolada, duas ou mais, pode não ser o suficiente e o pior, elas podem nos levar a cometer erros em nossas decisões, o gráfico de dispersão em 3D que montamos ele ajuda muito a entendermos como aquelas 3 variáveis se comportam para a definição dos melhores imóveis, porém, se a leitura dele pode ser difícil até para um analista fazer de bate e pronto, então imagina para uma diretoria...

Então para o dashboard a minha ideia e traduzir aquele gráfico em uma tabela, que simplesmente traga um 'Top 5 Melhores Imóveis' e para isso, eu vou criar um sistema de pontuação que relacione essas 3 váriaveis.

!!! danger "Ponto de Atenção"

    Não podemos simplesmente relacionar um valor com o outro diretamente, que diferente de um gráfico, não teremos eixos distintos, trabalharemos em um único plano, então o valor do imóvel tem muito mais dígitos que comparado as notas que vão até o número 5.
!!! success "Solução:" 
    Normalizar os dados.

 
 
!!! danger "Ponto de Atenção" 
    As 3 variáveis tendo a mesma proporção traz uma nova questão, uma variável é mais importante que outra para caráter de decisão?  
!!! success "Solução:"
    Criação arbitrária de pesos, para dosar cada variável.



!!! info "Fórmula:"
    Score = (Peso_Preço * Preço) + (Peso_Nota * Nota) + (Peso_Qty_Review * Qty_Review)

``` py
# Defição dos pesos para cada variável
peso_preco = 0.4
peso_nota = 0.2
peso_reviews = 0.4
# Esses valores podem ser alterados, mas a soma de todos precisa resutar em 1


# Vamos criar um novo DF copiando o DF original
score_df = out_df.copy()

# Normalizando as colunas 'price_usd', 'review_scores_rating' e 'qty_reviews'
scaler = MinMaxScaler()

# Essas colunas são auxiliares, então após o cálculo da pontuação, podemos excluí-las
score_df[['price_usd_normalized', 'review_scores_rating_normalized', 'qty_reviews_normalized']] = scaler.fit_transform(out_df[['price_usd', 'review_scores_rating', 'qty_reviews']])

# Calcular a pontuação usando a fórmula
score_df['Score'] = (peso_preco * score_df['price_usd_normalized'] +
                             peso_nota * score_df['review_scores_rating_normalized'] +
                             peso_reviews * score_df['qty_reviews_normalized'])

# Excluindo as colunas normalizadas
score_df.drop(['price_usd_normalized', 'review_scores_rating_normalized', 'qty_reviews_normalized'], axis=1, inplace=True)

# Ordenando o DataFrame pela pontuação em ordem decrescente
score_df = score_df.sort_values(by='Score', ascending=False)

score_df.head()

```
Vamos agora dar uma analisada neste Top 5.

Embora tenhamos preços muito altos no topo, também temos em algumas posições valores bem baixos, vamos multiplicá-los pela quantidade de reviews, na ideia que isso proporcionalmente indique a quantidade de clientes destes imóveis e assim, termos um valor arrecadado total com o imóvel.

!!! info "Detalhe:" 
    Esse valor arrecadado não é o valor absoluto arrecadado, porque muitas pessoas alugam os imóveis e não enviam review, este é um valor tendo base apenas a quantidade de pessoas que sabemos devido as reviews.

``` py
novo_df = score_df[['price_usd', 'review_scores_rating', 'qty_reviews','Score']].copy()
novo_df['price_accumulated'] = score_df['price_usd'] * score_df['qty_reviews']
novo_df.head()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_usd</th>
      <th>review_scores_rating</th>
      <th>qty_reviews</th>
      <th>Score</th>
      <th>price_accumulated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>235968</th>
      <td>26381.74</td>
      <td>4.75</td>
      <td>97.0</td>
      <td>0.596239</td>
      <td>2559028.78</td>
    </tr>
    <tr>
      <th>167067</th>
      <td>82.71</td>
      <td>4.70</td>
      <td>891.0</td>
      <td>0.589137</td>
      <td>73694.61</td>
    </tr>
    <tr>
      <th>131663</th>
      <td>80.44</td>
      <td>4.80</td>
      <td>828.0</td>
      <td>0.564823</td>
      <td>66604.32</td>
    </tr>
    <tr>
      <th>239720</th>
      <td>26381.74</td>
      <td>4.65</td>
      <td>33.0</td>
      <td>0.563507</td>
      <td>870597.42</td>
    </tr>
    <tr>
      <th>178743</th>
      <td>9357.35</td>
      <td>4.75</td>
      <td>526.0</td>
      <td>0.554783</td>
      <td>4921966.10</td>
    </tr>
  </tbody>
</table>
</div>

Mesmo o preço acumulado dos imóveis de 80 dólares sendo inferior ao #4 e #5, ainda veja como merecido a posição deles, aquele número alto de reviews com mais de 800 me diz mais sobre o sucesso daqueles imóveis do que o que teve 33 reviews.

Eu testei várias proporções de pesos, mas a que achei mais interessante foi a relação de 80% do pseudo preço acumulado + 20% de ponderação pela nota geral das reviews.

Você pode testar outras, no dashboard teremos a tabela dos melhores, junto com o ID do imóvel, assim bou fazer outro gráfico que seja possível acompanhar a quantidade de reviews históricas de um imóvel em específico, trazendo ainda mais contexto para o nosso sistema de pontuação.

``` py
# Extrair a coluna 'host_location'
coluna_host_location = df['host_location']

# Criar um novo DataFrame apenas com a coluna 'host_location'
novo_df = pd.DataFrame({'host_location': coluna_host_location})

# Salvar o novo DataFrame em um arquivo Excel
novo_df.to_excel('host_location.xlsx', index=False, engine='openpyxl')
```
Antes de sair eu baixei esta coluna só para dar uma analisada no Excel mesmo, tenho 2 pontos para ela:

* Incluir lista suspensa para validação de dados para a localização;
* Resolver o problema de 2 ou mais países cadastrados em um único registro.

Essa coluna foi excluída da nossa base final, mas pensando como um profissional da companhia, este ponto não poderia ser negligenciado, assim como tornar obrigatórios alguns campos como a quantidade de quartos.

## Salvando os Resultados 💾

``` py
output = 'Listings_output.csv'

# Salve o DataFrame como um arquivo CSV com codificação UTF-8
score_df.to_csv(output, encoding='utf-8', index=False)

print(f'O arquivo {output} foi salvo com sucesso.')

output = 'Reviews_output.csv'

# Salve o DataFrame como um arquivo CSV com codificação UTF-8
df_rev.to_csv(output, encoding='utf-8', index=False)

print(f'O arquivo {output} foi salvo com sucesso.')
```

