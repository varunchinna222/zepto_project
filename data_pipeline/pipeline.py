from pathlib import Path
import sqlite3, requests, pandas as pd
from bs4 import BeautifulSoup

print("Capstone")

BASE='https://books.toscrape.com/'
RATE=105.50
ROOT=Path(__file__).resolve().parent
DB=ROOT/'books.db'
OUT=ROOT/'sql_outputs.md'

def scrape(pages=5):
    rows=[]; url=BASE+'index.html'
    for page in range(pages):
        r=requests.get(url,timeout=20); r.raise_for_status()
        soup=BeautifulSoup(r.text,'html.parser')
        for card in soup.select('article.product_pod'):
            title=card.h3.a.get('title','').strip()
            price=card.select_one('.price_color').get_text(strip=True)
            rating_cls=card.select_one('.star-rating').get('class',[])
            rating_text=next((x for x in rating_cls if x in {'One','Two','Three','Four','Five'}),'')
            availability=card.select_one('.availability').get_text(' ',strip=True)
            # product cards expose category only through the product detail page
            href=card.h3.a.get('href')
            detail=requests.get(requests.compat.urljoin(url,href),timeout=20); detail.raise_for_status()
            ds=BeautifulSoup(detail.text,'html.parser')
            crumbs=[x.get_text(strip=True) for x in ds.select('ul.breadcrumb li')]
            category=crumbs[-2] if len(crumbs)>=2 else 'Unknown'
            rows.append(dict(title=title,price=price,star_rating=rating_text,availability=availability,category=category))
        nxt=soup.select_one('li.next a')
        if not nxt: break
        url=requests.compat.urljoin(url,nxt.get('href'))
    return pd.DataFrame(rows)

def clean(df):
    rating_map={'One':1,'Two':2,'Three':3,'Four':4,'Five':5}
    df['price_gbp']=pd.to_numeric(df['price'].str.replace('£','',regex=False),errors='coerce')
    df['rating']=df['star_rating'].map(rating_map)
    df['in_stock']=df['availability'].str.contains('In stock',case=False,na=False)
    for c in ['price_gbp','rating']:
        df[c]=df[c].fillna(df[c].median())
    df=df.dropna(subset=['title','category']).copy()
    df['rating']=df['rating'].astype(int)
    df['in_stock']=df['in_stock'].astype(bool)
    df['price_inr']=(df['price_gbp']*RATE).round(2)
    return df[['title','price_gbp','price_inr','rating','in_stock','category']]

def load_and_query(df):
    if DB.exists(): DB.unlink()
    con=sqlite3.connect(DB); con.execute('PRAGMA foreign_keys=ON')
    con.executescript('''CREATE TABLE categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE NOT NULL); CREATE TABLE books(book_id INTEGER PRIMARY KEY, title TEXT NOT NULL, price_gbp REAL NOT NULL, price_inr REAL NOT NULL, rating INTEGER NOT NULL, in_stock INTEGER NOT NULL, category_id INTEGER NOT NULL REFERENCES categories(category_id));''')
    cats=pd.DataFrame({'category_name':sorted(df.category.unique())}); cats.to_sql('categories',con,if_exists='append',index=False)
    mapping=dict(con.execute('SELECT category_name,category_id FROM categories').fetchall())
    load=df.copy(); load['category_id']=load.category.map(mapping); load[['title','price_gbp','price_inr','rating','in_stock','category_id']].to_sql('books',con,if_exists='append',index=False)
    queries={
      '1 SELECT WHERE':"SELECT title, price_inr, rating FROM books WHERE in_stock=1 AND price_inr BETWEEN 500 AND 1500 ORDER BY price_inr DESC LIMIT 10;",
      '2 ORDER BY LIMIT':"SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 10;",
      '3 DISTINCT':"SELECT DISTINCT category_name FROM categories ORDER BY category_name;",
      '4 IN':"SELECT title, rating FROM books WHERE rating IN (4,5) ORDER BY rating DESC, title LIMIT 10;",
      '5 JOIN':"SELECT c.category_name, b.title, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;"
    }
    lines=['# SQL query outputs','']
    results={}
    for name,q in queries.items():
        out=pd.read_sql(q,con); results[name]=out
        lines += [f'## {name}', '```sql', q, '```', '', out.to_markdown(index=False), '']
    # pandas read_sql and merge reproduction
    sql_join=results['5 JOIN']
    books_mem=load.copy(); cats_mem=cats.copy(); cats_mem['category_id']=cats_mem['category_name'].map(mapping)
    merged=books_mem.merge(cats_mem,on=['category_id','category'],how='inner')
    merged=merged[['category_name','title','rating','price_inr']].sort_values(['rating','price_inr'],ascending=[False,False]).head(10).reset_index(drop=True)
    sql_norm=sql_join.reset_index(drop=True)
    lines += ['## pandas validation','`pd.read_sql` JOIN output:', '', sql_norm.to_markdown(index=False),'','`pd.merge` JOIN output:', '', merged.to_markdown(index=False),'',f'Equivalent: **{sql_norm.equals(merged)}**.']
    OUT.write_text('\n'.join(lines),encoding='utf-8'); con.close(); return results

if __name__=='__main__':
    raw=scrape(5)
    raw.to_csv(ROOT/'raw_scraped.csv',index=False)
    clean_df=clean(raw)
    clean_df.to_csv(ROOT/'cleaned_books.csv',index=False)
    assert len(clean_df)>=60 and clean_df.category.nunique()>=3
    load_and_query(clean_df)
    print(f'Scraped {len(clean_df)} books across {clean_df.category.nunique()} categories.')
