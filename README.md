# Redis-search

# Building a search engine using Redis and redis-py

For those of you who aren't quite so caught up in the recent happenings in the open source server software world, Redis is a remote data structure server.  You can think of it like memcached with strings, lists, sets, hashes, and zsets (hashes that you can sort by value).  All of the operations that you expect are available (list push/pop from either end, sorting lists and sets, sorting based on a lookup key/hash, ...), and some that you wouldn't expect (set intersection/union, zset intersection/union with 3 aggregation methods, ...).  Throw in master/slave replication, on-disk persistence, clients for most major modern languages, a fairly active discussion group to help you as necessary, and you can have a valuable new piece of infrastructure for free.

I know what you are thinking.  Why would we want to build a search engine from scratch when Lucene, Xapian, and other software is available?  What could possibly be gained?  To start, simplicity, speed, and flexibility.  We're going to be building a search engine implementing TF/IDF search Redis, redis-py, and just a few lines of Python.  With a few small changes to what I provide, you can integrate your own document importance scoring, and if one of my patches gets merged into Redis, you could combine TF/IDF with your pre-computed Pagerank... Building an index and search engine using Redis offers so much more flexibility out of the box than is available using any of the provided options.  Convinced?


First thing's first, you need to have a recent version of Redis installed on your platform.  Until 2.0 is released, you're going to need to use git head, as we'll be using some features that were not available in a "stable" release (though I've used the 1.3.x series for months).  After Redis is up and running, go ahead and install redis-py.


If you haven't already done so, have a read of another great post on doing fuzzy full-text search using redis and Python over on PlayNice.ly's blog.  They use metaphone/double metaphone to extract how a word sounds, which is a method of pre-processing to handle spelling mistakes.  The drawback to the metaphone algorithms is that it can be overzealous in it's processing, and can result in poor precision.  In the past I've used the Porter Stemming algorithm to handle tense normalization (jump, jumping, jumped all become jump).  Depending on the context, you can use one, neither, or both to improve search quality.  For example, in the context of machine learning, metaphone tends to remove too many features from your documents to make LSI or LDA clustering worthwhile, though stemming actually helps with clustering.  We aren't going to use either of them for the sake of simplicity here, but the source will point out where you can add either or both of them in order to offer those features.

Have everything up and running?  Great.  Let's run some tests...
```
>>> import redis
>>> r = redis.Redis()
>>> r.sadd('temp1', '1')
True
>>> r.sadd('temp2', '2')
True
>>> r.sunion(['temp1', 'temp2'])
set(['1', '2'])
>>> p = r.pipeline()
>>> r.scard('temp1')
1
>>> p.scard('temp1')
<redis.client.pipeline object at 0x022EC420>
>>> p.scard('temp2')
<redis.client.Pipeline object at 0x022EC420>
>>> p.execute()
[1, 1]
>>> r.zunionstore('temp3', {'temp1':2, 'temp2':3})
2
>>> r.zrange('temp3', 0, -1, withscores=True)
[('1', 2.0), ('2', 3.0)]
```

Believe it or not, that's more or less the meat of everything that we're going to be using.  We add items to sets, union some sets with weights, use pipelines to minimize our round-trips, and pull the items out with scores.  Of course the devil is in the details.


The first thing we need to do in order to index documents is to parse them.  What works fairly well as a start is to only include alpha-numeric characters.  I like to throw in apostrophies for contractions like "can't", "won't", etc.  If you use the Porter Stemmer or Metaphone, contractions and ownerships (like Joe's) can be handled automatically.  Pro tip: if you use stemming, don't be afraid to augment your stemming with a secondary word dictionary to ensure that what the stemmer produces is an actual base word.


In our case, because indexing and index removal are so similar, we're going to overload a few of our functions to do slightly different things, depending on the context.  We'll use the simple parser below as a start...

```
import re
 
NON_WORDS = re.compile("[^a-z0-9' ]")
 
# stop words pulled from the below url
# http://www.textfixer.com/resources/common-english-words.txt

STOP_WORDS = set('''a able about across after all almost also am
among an and any are as at be because been but by can cannot
could dear did do does either else ever every for from get got
had has have he her hers him his how however i if in into is it
its just least let like likely may me might most must my neither
no nor not of off often on only or other our own rather said say
says she should since so some than that the their them then
there these they this tis to too twas us wants was we were what
when where which while who whom why will with would yet you
your'''.split())
 
def get_index_keys(content, add=True):
    # Very simple word-based parser.  We skip stop words and
    # single character words.
    words = NON_WORDS.sub(' ', content.lower()).split()
    words = [word.strip("'") for word in words]
    words = [word for word in words
                if word not in STOP_WORDS and len(word) > 1]
    # Apply the Porter Stemmer here if you would like that
    # functionality.
 
    # Apply the Metaphone/Double Metaphone algorithm by itself,
    # or after the Porter Stemmer.
 
    if not add:
        return words
 
    # Calculate the TF portion of TF/IDF.
    counts = collections.defaultdict(float)
    for word in words:
        counts[word] += 1
    wordcount = len(words)
    tf = dict((word, count / wordcount)
                for word, count in counts.iteritems())
    return tf
```

In document search/retrieval, stop words are those words that are so common as to be mostly worthless to indexing or search.  The set of common words provided is a little aggressive, but it also helps to keep searches directed to the content that is important.


In your own code, feel free to tweak the parsing to suit your needs.  Phrase parsing, url extraction, hash tags, @tags, etc., are all very simple and useful additions that can improve searching quality on a variety of different types of data.  In particular, don't be afraid to create special tokens to signify special cases, like "has_url" or "has_attachment" for email indexes, "is_banned" or "is_active" for user searches.


Now that we have parsing, we merely need to add our term frequencies to the proper redis zsets.  Just like getting our keys to index, adding and removing from the index are almost identical, so we'll be using the same function for both tasks...

```
def handle_content(connection, prefix, id, content, add=True):
    # Get the keys we want to index.
    keys = get_index_keys(content)
 
    # Use a non-transactional pipeline here to improve
    # performance.
    pipe = connection.pipeline(False)
 
    # Since adding and removing items are exactly the same,
    # except for the method used on the pipeline, we will reduce
    # our line count.
    if add:
        pipe.sadd(prefix + 'indexed:', id)
        for key, value in keys.iteritems():
            pipe.zadd(prefix + key, id, value)
    else:
        pipe.srem(prefix + 'indexed:', id)
        for key in keys:
            pipe.zrem(prefix + key, id)
 
    # Execute the insertion/removal.
    pipe.execute()
 
    # Return the number of keys added/removed.
    return len(keys)
```

In Redis, pipelines allow for the bulk execution of commands in order to reduce the number of round-trips, optionally including non-locking transactions (a transaction will fail if someone modifies keys that you are watching; see the Redis wiki on it's semantics and use).  For Redis, fewer round-trips translate into improved performance, as the slow part of most Redis interactions is network latency.


The entirety of the above handle_content() function basically just added or removed some zset key/value pairs.  At this point we've indexed our data.  The only thing left is to search...

```
import math
import os
 
def search(connection, prefix, query_string, offset=0, count=10):
    # Get our search terms just like we did earlier...
    keys = [prefix + key
            for key in get_index_keys(query_string, False)]
 
    if not keys:
        return [], 0
 
    total_docs = max(
        connection.scard(prefix + 'indexed:'), 1)
 
    # Get our document frequency values...
    pipe = self.connection.pipeline(False)
    for key in keys:
        pipe.zcard(key)
    sizes = pipe.execute()
 
    # Calculate the inverse document frequencies...
    def idf(count):
        # Calculate the IDF for this particular count
        if not count:
            return 0
        return max(math.log(total_docs / count, 2), 0)
    idfs = map(idf, sizes)
 
    # And generate the weight dictionary for passing to
    # zunionstore.
    weights = dict((key, idfv)
            for key, size, idfv in zip(keys, sizes, idfs)
                if size)
 
    if not weights:
        return [], 0
 
    # Generate a temporary result storage key
    temp_key = prefix + 'temp:' + os.urandom(8).encode('hex')
    try:
        # Actually perform the union to combine the scores.
        known = connection.zunionstore(temp_key, weights)
        # Get the results.
        ids = connection.zrevrange(
            temp_key, offset, offset+count-1, withscores=True)
    finally:
        # Clean up after ourselves.
        self.connection.delete(temp_key)
    return ids, known
```

Breaking it down, the first part parses the search terms the same way as we did during indexing.  The second part fetches the number of documents that have that particular word, which is necessary for the IDF portion of TF/IDF.  The third part calculates the IDF, packing it into a weights dictionary.  Then finally, we use the ZUNIONSTORE command to take individual TF scores for a given term, multiply them by the IDF for the given term, then combine based on the document id and return the highest scores.  And that's it.


No, really.  Those snippets are all it takes to build a working and functional search engine using Redis.  I've gone ahead and tweaked the included snippets to offer a more useful interface, as well as a super-minimal test case.  You can find it as this Github Gist.


#### A few ideas for tweaks/improvements:
You can replace the TF portion of TF/IDF with the constant 1.  Doing so allows us to replace the zset document lists with standard sets, which will reduce Redis' memory requirements significantly for large indexes.  Depending on the documents you are indexing/searching, this can reduce or improve the quality of search results significantly.  Don't be afraid to test both ways.

Search quality on your personal site is all about parsing.Parse your documents so that your users can find them in a variety of ways.  As stated earlier: @tags, #tags, ^references (for twitter/social-web like experiences), phrases, incoming/outgoing urls, etc.

Parse your search queries in an intelligent way, and do useful things with it.  If someone provides "web history search +firefox -ie" as a search query, boost the IDF for the "firefox" term and make the IDF negative for the "ie" term.  If you have tokens like "has_url", then look for that as part of the search query.

If you are using the TF weight as 1, and have used sets, you can use the SDIFF command to explicitly exclude those sets of documents with the -negated terms.

There are three commands in search that are executed outside of a pipeline.  The first one can be merged into the pipeline just after, but you'll have to do some slicing.  The ZUNIONSTORE and ZRANGE calls can be combined into another pipeline, though their results need to be reversed with respect to what the function currently returns.

You can store all of the keys indexed for a particular document id in a set.  Un-indexing any document then can be performed by fetching the set names via SMEMBERS, followed by the relevant ZREM calls, the one 'indexed' SREM call, and the deletion of the set that contained all of the indexed keys.  Also, if you get an index call for a document that is already indexed, you can either un-index and re-index, or you can return early.  It's up to you to determine the semantics you want.


There are countless improvements that can be done to this basic index/search code.  Don't be afraid to try different ideas to see what you can build.


## Warning
Using Redis to build search is great for your personal site, your company intranet, your internal customer search, maybe even one of your core products.  But be aware that Redis keeps everything in memory, so as your index grows, so does your machine requirements.  Naive sharding tricks may work to a point, but there will be a point where your merging will have to turn into a tree, and your layers of merges start increasing your latency to scary levels.
