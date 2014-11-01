
###################################################
### chunk number 1: Preliminaries
###################################################

library("clue")
library("tm")

###################################################
### chunk number 2: NewCorpus eval=FALSE
###################################################
## new("Corpus", .Data = ..., DMetaData = ..., CMetaData = ..., DBControl = ...)


###################################################
### chunk number 3: CorpusConstructor eval=FALSE
###################################################
## Corpus(object = ...,
##            readerControl = list(reader = object@DefaultReader,
##                            language = "en_US",
##                            load = FALSE),
##            dbControl = list(useDb = TRUE,
##                             dbName = "texts.db",
##                             dbType = "DB1"))


###################################################
### chunk number 4: NewPlainTextDocument eval=FALSE
###################################################
## new("PlainTextDocument", .Data = "Some text.", URI = uri, Cached = TRUE,
##     Author = "Mr. Nobody", DateTimeStamp = Sys.time(),
##     Description = "Example", ID = "ID1", Origin = "Custom",
##     Heading = "Ex. 1", Language = "en_US")


###################################################
### chunk number 5: NewTextRepository eval=FALSE
###################################################
## new("TextRepository",
##     .Data = list(Col1, Col2), RepoMetaData = list(created = "now"))


###################################################
### chunk number 6: NewTermDocMatrix eval=FALSE
###################################################
## new("TermDocMatrix", Data = tdm, Weighting = weightTf)


###################################################
### chunk number 7: WekaTokenizer eval=FALSE
###################################################
## TermDocMatrix(col, control = list(tokenize = NGramTokenizer))


###################################################
### chunk number 8: openNLPTokenizer eval=FALSE
###################################################
## TermDocMatrix(col, control = list(tokenize = tokenize))


###################################################
### chunk number 9: SentenceDetection eval=FALSE
###################################################
## TermDocMatrix(col, control = list(tokenize = sentDetect))


###################################################
### chunk number 10: NewDirSource eval=FALSE
###################################################
## new("DirSource", LoDSupport = TRUE, FileList = dir(),
##     Position = 0, DefaultReader = readPlain, Encoding = "latin1")


###################################################
### chunk number 11: Ovid
###################################################
txt <- system.file("texts", "txt", package = "tm")
(ovid <- Corpus(DirSource(txt),
                readerControl = list(reader = readPlain,
                                     language = "la",
                                     load = TRUE)))


###################################################
### chunk number 12: CorpusDBSupport eval=FALSE
###################################################
## Corpus(DirSource(txt),
##            readerControl = list(reader = readPlain,
##                                 language = "la", load = TRUE),
##            dbControl = list(useDb = TRUE,
##                             dbName = "/home/user/oviddb",
##                             dbType = "DB1"))


###################################################
### chunk number 13: IDOvid
###################################################
ID(ovid[[1]])


###################################################
### chunk number 14: AuthorOvid
###################################################
Author(ovid[[1]]) <- "Publius Ovidius Naso"


###################################################
### chunk number 15: 
###################################################
meta(ovid[[1]])


###################################################
### chunk number 16: OvidSubset
###################################################
ovid[1:3]


###################################################
### chunk number 17: OvidDocument
###################################################
ovid[[1]]


###################################################
### chunk number 18: Concatenation
###################################################
c(ovid[1:2], ovid[3:4])


###################################################
### chunk number 19: LengthOvid
###################################################
length(ovid)


###################################################
### chunk number 20: SummaryOvid
###################################################
summary(ovid)


###################################################
### chunk number 21: tmUpdate
###################################################
tmUpdate(ovid, DirSource(txt))


###################################################
### chunk number 22: OvidMeta
###################################################
ovid <- appendMeta(ovid,
                   cmeta = list(test = c(1,2,3)),
                   dmeta = list(clust = c(1,1,2,2,2)))
summary(ovid)
CMetaData(ovid)
DMetaData(ovid)


###################################################
### chunk number 23: OvidAppendElem
###################################################
(ovid <- appendElem(ovid, data = ovid[[1]], list(clust = 1)))


###################################################
### chunk number 24: TextRepo
###################################################
(repo <- TextRepository(ovid))


###################################################
### chunk number 25: TextRepoMeta
###################################################
repo <- appendElem(repo, ovid, list(modified = date()))
repo <- appendMeta(repo, list(moremeta = 5:10))
summary(repo)
RepoMetaData(repo)


###################################################
### chunk number 26: 
###################################################
meta(ovid, type = "corpus", "foo") <- "bar"
meta(ovid, type = "corpus")
meta(ovid, "someTag") <- 6:11
meta(ovid)


###################################################
### chunk number 27: tmMap
###################################################
tmMap(ovid, FUN = tmTolower)


###################################################
### chunk number 28: tmFilter
###################################################
tmFilter(ovid, FUN = searchFullText, "Venus", doclevel = TRUE)


###################################################
### chunk number 29: tmIndex
###################################################
tmIndex(ovid, "identifier == '2'")


###################################################
### chunk number 30: GmaneSource eval=FALSE
###################################################
## setClass("GmaneSource",
##          representation(URI = "ANY", Content = "list"),
##          contains = c("Source"))


###################################################
### chunk number 31: GmaneSourceConstructor eval=FALSE
###################################################
## setMethod("GmaneSource",
##           signature(object = "ANY"),
##           function(object, encoding = "UTF-8") {
##               ## ---code chunk---
##               new("GmaneSource", LoDSupport = FALSE, URI = object,
##                   Content = content, Position = 0, Encoding = encoding)
##           })


###################################################
### chunk number 32: stepNext eval=FALSE
###################################################
## setMethod("stepNext",
##           signature(object = "GmaneSource"),
##           function(object) {
##               object@Position <- object@Position + 1
##               object
##           })


###################################################
### chunk number 33: getElem eval=FALSE
###################################################
## setMethod("getElem",
##           signature(object = "GmaneSource"),
##           function(object) {
##               ## ---code chunk---
##               list(content = content, uri = object@URI)
##           })


###################################################
### chunk number 34: eoi eval=FALSE
###################################################
## setMethod("eoi",
##           signature(object = "GmaneSource"),
##           function(object) {
##               length(object@Content) <= object@Position
##           })


###################################################
### chunk number 35: readGmane eval=FALSE
###################################################
## readGmane <- FunctionGenerator(function(...) {
##     function(elem, load, language, id) {
##         ## ---code chunk---
##         new("NewsgroupDocument", .Data = content, URI = elem$uri,
##             Cached = TRUE, Author = author, DateTimeStamp = datetimestamp,
##             Description = "", ID = id, Origin = origin, Heading = heading,
##             Language = language, Newsgroup = newsgroup)
##     }
## })


###################################################
### chunk number 36: GmaneCorpus
###################################################
rss <- system.file("texts", "gmane.comp.lang.r.gr.rdf", package = "tm")
Corpus(GmaneSource(rss), readerControl = list(reader = readGmane, language = "en_US", load = TRUE))


###################################################
### chunk number 37: readPDF eval=FALSE
###################################################
## readPDF <- FunctionGenerator(function(...) {
##   function(elem, load, language, id) {
##     ## get metadata
##     meta <- system(paste("pdfinfo", as.character(elem$uri[2])),
##                    intern = TRUE)
## 
##     ## extract and store main information, e.g.:
##     heading <- gsub("Title:[[:space:]]*", "",
##                     grep("Title:", meta, value = TRUE))
## 
##     ## [... similar for other metadata ...]
## 
##     ## extract text from PDF using the external pdftotext utility:
##     corpus <- paste(system(paste("pdftotext", as.character(elem$uri[2]), "-"),
##                            intern = TRUE),
##                     sep = "\n", collapse = "")
## 
##     ## create new text document object:
##     new("PlainTextDocument", .Data = corpus, URI = elem$uri, Cached = TRUE,
##         Author = author, DateTimeStamp = datetimestamp,
##         Description = description, ID = id, Origin = origin,
##         Heading = heading, Language = language)
##     }
## })


###################################################
### chunk number 38: TransformExtension eval=FALSE
###################################################
## setGeneric("myTransform", function(object, ...) standardGeneric("myTransform"))
## setMethod("myTransform",
##           signature(object = "PlainTextDocument"),
##           function(object, ..., DMetaData) {
##               Content(object) <- doSomeThing(object, DMetaData)
##               return(object)
##           })


###################################################
### chunk number 39: Reuters
###################################################
reut21578XMLgz <- system.file("texts", "reut21578.xml.gz", package = "tm")
(Reuters <- Corpus(ReutersSource(gzfile(reut21578XMLgz)),
                   readerControl = list(reader = readReut21578XML,
                                        language = "en_US",
                                        load = TRUE)))


###################################################
### chunk number 40: 
###################################################
tmMap(Reuters, asPlain)


###################################################
### chunk number 41:  eval=FALSE
###################################################
## tmFilter(crude, "Topics == 'crude'")


###################################################
### chunk number 42: AcqCrude
###################################################
data("acq")
data("crude")


###################################################
### chunk number 43:  eval=FALSE
###################################################
## acq[[10]]


###################################################
### chunk number 44: 
###################################################
strwrap(acq[[10]])


###################################################
### chunk number 45:  eval=FALSE
###################################################
## stemDoc(acq[[10]])


###################################################
### chunk number 46: 
###################################################
strwrap(stemDoc(acq[[10]]))


###################################################
### chunk number 47: 
###################################################
tmMap(acq, stemDoc)


###################################################
### chunk number 48:  eval=FALSE
###################################################
## stripWhitespace(acq[[10]])


###################################################
### chunk number 49: 
###################################################
strwrap(stripWhitespace(acq[[10]]))


###################################################
### chunk number 50:  eval=FALSE
###################################################
## tmTolower(acq[[10]])


###################################################
### chunk number 51: 
###################################################
strwrap(tmTolower(acq[[10]]))


###################################################
### chunk number 52: stopwords
###################################################
mystopwords <- c("and", "for", "in", "is", "it", "not", "the", "to")


###################################################
### chunk number 53:  eval=FALSE
###################################################
## removeWords(acq[[10]], mystopwords)


###################################################
### chunk number 54: 
###################################################
strwrap(removeWords(acq[[10]], mystopwords))


###################################################
### chunk number 55:  eval=FALSE
###################################################
## tmMap(acq, removeWords, mystopwords)


###################################################
### chunk number 56:  eval=FALSE
###################################################
## stopwords(language = ...)


###################################################
### chunk number 57: 
###################################################
library("wordnet")


###################################################
### chunk number 58: 
###################################################
synonyms("company")


###################################################
### chunk number 59:  eval=FALSE
###################################################
## replaceWords(acq[[10]], synonyms(dict, "company"), by = "company")


###################################################
### chunk number 60:  eval=FALSE
###################################################
## tmMap(acq, replaceWords, synonyms(dict, "company"), by = "company")


###################################################
### chunk number 61:  eval=FALSE
###################################################
## library("openNLP")
## tagPOS(acq[[10]])


###################################################
### chunk number 62: 
###################################################
library("openNLP")
strwrap(tagPOS(acq[[10]]))


###################################################
### chunk number 63: 
###################################################
# Creates the term-document matrix for our crude data set
crudeTDM <- TermDocMatrix(crude, control = list(stopwords = TRUE))


###################################################
### chunk number 64: 
###################################################
# Terms with more than 10 occurrences
(crudeTDMHighFreq <- findFreqTerms(crudeTDM, 10, Inf))


###################################################
### chunk number 65: 
###################################################
# Frequencies for high frequency terms
Data(crudeTDM)[1:10,crudeTDMHighFreq]


###################################################
### chunk number 66: 
###################################################
findAssocs(crudeTDM, "oil", 0.85)


###################################################
### chunk number 67: 
###################################################
plot(crudeTDM, corThreshold = 0.5, terms = findFreqTerms(crudeTDM, 6, Inf))


###################################################
### chunk number 68: 
###################################################
ws <- c(acq, crude)
summary(ws)


###################################################
### chunk number 69: dissimilarityTermDocMatrix eval=FALSE
###################################################
## dissimilarity(crudeTDM, method = "cosine")


###################################################
### chunk number 70: dissimilarityTwoDocs
###################################################
dissimilarity(crude[[1]], crude[[2]], "cosine")


###################################################
### chunk number 71: wsTermDocMatrix
###################################################
wsTDM <- Data(TermDocMatrix(ws))


###################################################
### chunk number 72: 
###################################################
wsHClust <- hclust(dist(wsTDM), method = "ward")


###################################################
### chunk number 73: 
###################################################
plot(wsHClust, labels = c(rep("acq",50), rep("crude",20)))


###################################################
### chunk number 74: kmeans
###################################################
wsKMeans <- kmeans(wsTDM, 2)


###################################################
### chunk number 75: 
###################################################
wsReutersCluster <- c(rep("acq",50), rep("crude",20))


###################################################
### chunk number 76: 
###################################################
cl_agreement(wsKMeans, as.cl_partition(wsReutersCluster), "diag")


###################################################
### chunk number 77: 
###################################################
library("class")
library("kernlab")
data(spam)


###################################################
### chunk number 78: 
###################################################
train <- rbind(spam[1:1360, ], spam[1814:3905, ])


###################################################
### chunk number 79: 
###################################################
trainCl <- train[,"type"]


###################################################
### chunk number 80: 
###################################################
test <- rbind(spam[1361:1813, ], spam[3906:4601, ])


###################################################
### chunk number 81: 
###################################################
trueCl <- test[,"type"]


###################################################
### chunk number 82: knn
###################################################
knnCl <- knn(train[,-58], test[,-58], trainCl)


###################################################
### chunk number 83: 
###################################################
(nnTable <- table("1-NN" = knnCl, "Reuters" = trueCl))


###################################################
### chunk number 84: 
###################################################
sum(diag(nnTable))/nrow(test)


###################################################
### chunk number 85: ksvm
###################################################
ksvmTrain <- ksvm(type ~ ., data = train)


###################################################
### chunk number 86: 
###################################################
svmCl <- predict(ksvmTrain, test[,-58])


###################################################
### chunk number 87: 
###################################################
(svmTable <- table("SVM" = svmCl, "Reuters" = trueCl))


###################################################
### chunk number 88: 
###################################################
sum(diag(svmTable))/nrow(test)


###################################################
### chunk number 89: 
###################################################
# Instantiate both string kernels
stringkern <- stringdot(type = "string")


###################################################
### chunk number 90:  eval=FALSE
###################################################
## # Perform spectral clustering with string kernels
## stringCl <- specc(ws, 2, kernel = stringkern)


###################################################
### chunk number 91: specc
###################################################
set.seed(1234)
stringkern
stringCl <- as.vector(specc(ws, 2, kernel = stringkern))


###################################################
### chunk number 92: 
###################################################
table("String Kernel" = stringCl, "Reuters" = wsReutersCluster)


###################################################
### chunk number 93:  eval=FALSE
###################################################
## convertMboxEml("2006.txt", "2006/")


###################################################
### chunk number 94: Rdevel eval=FALSE
###################################################
## rdevel <- Corpus(DirSource("2006/"),
##                      readerControl = list(reader = readNewsgroup,
##                                           language = "en_US",
##                                           load = TRUE))


###################################################
### chunk number 95:  eval=FALSE
###################################################
## rdevel <- tmMap(rdevel, asPlain)


###################################################
### chunk number 96:  eval=FALSE
###################################################
## rdevel <- tmMap(rdevel, stripWhitespace)
## rdevel <- tmMap(rdevel, tmTolower)


###################################################
### chunk number 97:  eval=FALSE
###################################################
## summary(rdevel)


###################################################
### chunk number 98:  eval=FALSE
###################################################
## tdm <- TermDocMatrix(rdevel, list(stemming = TRUE, stopwords = TRUE))


###################################################
### chunk number 99:  eval=FALSE
###################################################
## authors <- lapply(rdevel, Author)
## authors <- sapply(authors, paste, collapse = " ")


###################################################
### chunk number 100:  eval=FALSE
###################################################
## sort(table(authors), decreasing = TRUE)[1:3]


###################################################
### chunk number 101:  eval=FALSE
###################################################
## headings <- lapply(rdevel, Heading)
## headings <- sapply(headings, paste, collapse = " ")


###################################################
### chunk number 102:  eval=FALSE
###################################################
## (bigTopicsTable <- sort(table(headings), decreasing = TRUE)[1:3])
## bigTopics <- names(bigTopicsTable)


###################################################
### chunk number 103:  eval=FALSE
###################################################
## topicCol <- rdevel[headings == bigTopics[1]]
## unique(sapply(topicCol, Author))


###################################################
### chunk number 104:  eval=FALSE
###################################################
## topicCol <- rdevel[headings == bigTopics[2]]
## unique(sapply(topicCol, Author))


###################################################
### chunk number 105: bugCol eval=FALSE
###################################################
## (bugCol <- tmFilter(rdevel,
##                     FUN = searchFullText, "[^[:alpha:]]+bug[^[:alpha:]]+",
##                     doclevel = TRUE))


###################################################
### chunk number 106:  eval=FALSE
###################################################
## bugColAuthors <- lapply(bugCol, Author)
## bugColAuthors <- sapply(bugColAuthors, paste, collapse = " ")
## sort(table(bugColAuthors), decreasing = TRUE)[1:3]


###################################################
### chunk number 107: findFreqTermsTDM eval=FALSE
###################################################
## f <- findFreqTerms(tdm, 30, 31)
## sort(f[-grep("[0-9]", f)])


###################################################
### chunk number 108: removeCitationSignature
###################################################
setGeneric("removeCitationSignature",
           function(object, ...) standardGeneric("removeCitationSignature"))
setMethod("removeCitationSignature",
          signature(object = "PlainTextDocument"),
          function(object, ...) {
            c <- Content(object)
            
            ## Remove citations starting with '>'
            citations <- grep("^[[:blank:]]*>.*", c)
            if (length(citations) > 0)
              c <- c[-citations]
            
            ## Remove signatures starting with '-- '
            signatureStart <- grep("^-- $", c)
            if (length(signatureStart) > 0)
              c <- c[-(signatureStart:length(c))]
            
            Content(object) <- c
            return(object)
          })


###################################################
### chunk number 109:  eval=FALSE
###################################################
## rdevelInc <- tmMap(rdevel, removeCitationSignature)


###################################################
### chunk number 110:  eval=FALSE
###################################################
## tdmInc <- TermDocMatrix(rdevelInc, list(stemming = TRUE, stopwords = TRUE))


###################################################
### chunk number 111: findFreqTermsTDMInc eval=FALSE
###################################################
## f <- findFreqTerms(tdmInc, 30, 31)
## sort(f[-grep("[0-9]", f)])


###################################################
### chunk number 112: subjectCounts eval=FALSE
###################################################
## subjectCounts <- 0
## for (r in rdevelInc) {
##     ## Get single characters from subject
##     h <- unlist(strsplit(Heading(r), " "))
## 
##     ## Count unique matches of subject strings within document
##     len <- length(unique(unlist(lapply(h, grep, r, fixed = TRUE))))
## 
##     ## Update counter
##     subjectCounts <- c(subjectCounts, len)
## }
## summary(subjectCounts)



---
  title: "Text Mining Notes"
author: "Jas Sohi"
date: "Monday, October 27, 2014"
output: html_document
---
  
  *Document clustering - Google News categories*
  
  **Text Categorization - Email filter**
  
  **Packages**
  -kernlab
-tm
-lsa
-Rstem
-openNLP
-Wordnet
-kernlab
library(tm)

docs <- readDOC('ea.docx')

**Steps**
  1.Preprocess
- Create term-document matrix
- Stemming #stemDoc(acq[[10]])
- for all documents in the Corpus - tmMap(acq, stemDoc)
- Remove whitespace #stripWhitespace(acq[[10]])
- Conversion to lowercase # tmTolower(acq[[10]])
-  Stopword removal - create a list of stopwords - entropy is low (the amount of information they have)
- mystopwords <- c("and", "for", "in", "is", "it", "not", "the", "to")
- removeWords(acq[[10]], mystopwords)
- Whole collection - tmMap(acq, removeWords, mystopwords)
- All english stopwords - stopwords(language = english)
-Wordnet database package for synonyms
- Clustering - groups are not known before hand
- Classification - groups are known before hand

library('wordnet')


#matrix(data, nrow, ncol, byrow)
term-document matrix - distinct terms for each document

#Stemming 
-closing to close



#Data needs to be in a list?
data("crude")
crude$meta


tdm <- TermDocumentMatrix(crude)

t<- c(3,5)

<http://rmarkdown.rstudio.com>.

**Knit**
  
  ```{r}
summary(cars)
```

You can also embed plots, for example:
  
  ```{r, echo=FALSE}
plot(cars)
```

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.




libs <- c("tm","SnowballC","wordcloud","RColorBrewer", "openNLP")
lapply(libs, require, character.only = TRUE)

#txt3 <- tokenize()

directory <- "US"
setwd(directory)
getwd()
ovid  <- VCorpus(DirSource(directory), readerControl = list(language="lat")) #US is the foldername on WD
str(ovid[[3]])
ovid[[3]]$content

length(ovid[[3]]$content)
object.size(ovid[[3]])

inspect(ovid[1])
summary(ovid)

meta(ovid[[3]])
twitterdm <- TermDocumentMatrix(ovid)

Web-based so file size and access speed are crucial

#Are we supposed to know the length of the text file or read line by line?

#Read in file and create a sample from the dataset
#Used to create a smaller file to work with 
randomlines <- character()
set.seed(123) #the random binom the loop  
twittersmall <- 
  
  
  
  
  
  # while (length(oneLine <- readLines(readcon, 1, warn = FALSE)) > 0) { #length of a line is not empty
  #   if (rbinom(n=1,size=1,prob=0.2)==1){
  #     randomlines <- c(oneLine,randomlines)     
  #   }  
  # } 
  
close(readcon)
writecon <- file("output.txt")
writeLines(randomlines, writecon)
close(writecon)
###########################################

filename <- "en_US.twitter.txt"  
text2token <- function(filename){
  
  return(object) #tokenized version
}

con <- file("en_US.twitter.txt", "r")

#Profanity filter
#Source
#http://www.frontgatemedia.com/a-list-of-723-bad-words-to-blacklist-and-how-to-use-facebooks-moderation-tool/

profanity <- read.csv(file="profanity.csv", header=TRUE)
class(profanity)
nrow(profanity)

inspect(ovid[1])

data("crude")
class(crude) == class(ovid)

scan_tokenizer(crude[[1]])

scan_tokenizer(ovid[[3]])

tw <- readLines("en_US.twitter.txt") 
Encoding(tw) <- "UTF-8" #making sure, though my system is UTF-8
justUTF <-iconv(tw, "UTF-8", "UTF-8",sub='')
rm(tw)
lowtw <- tolower(justUTF)

memes <- data.frame(index= 1:length(tw))
memes$char <- nchar(tw)
memes$love <- 0
memes$love[grep("love", tw,value=FALSE)] <- 1
memes$hate <- 0
memes$hate[grep("hate", tw, perl=TRUE, value=FALSE)] <- 1
memes$chess <- 0
max(memes$char)
memes$chess[grep("A computer once beat me at chess, but it was no match for me at kickboxing", tw,value=FALSE)] <- 1

sum(memes$chess)

memes$biostats <- 0
word <- grep("biostats", tw,value=TRUE)
word

sum(memes$love)/sum(memes$hate)

Quiz #1
tw <- readLines("en_US/en_US.twitter.txt") 
blogs <- readLines("en_US.blogs.txt") 
news <- readLines("en_US.news.txt") 
bdf <- data.frame(index= 1:length(blogs))
ndf <- data.frame(index= 1:length(news))
bdf$char <- nchar(blogs)
ndf$char <- nchar(news)
max(ndf$char)
max(bdf$char)

# use readlines gsub replace profanity first and any other transformations before ceating corpus


# 
# Rgraphviz
# (Gentry et al.
#  , 2014) from the BioConductor repository for R (bioconductor.org) is
# used to plot the network graph that displays the correlation between chosen words in the corpus.
# Here we choose 50 of the more frequent words a

# 1. This is for english so we can ignore the other corpora?

# a <- tm_map(a, removeNumbers)
# a <- tm_map(a, removePunctuation)
# a <- tm_map(a , stripWhitespace)
# a <- tm_map(a, tolower)
# a <- tm_map(a, removeWords, stopwords("english")) # this stopword file is at C:\Users\[username]\Documents\R\win-library\2.13\tm\stopwords 
# a <- tm_map(a, stemDocument, language = "english")
#adtm <-DocumentTermMatrix(a) 
#adtm <- removeSparseTerms(adtm, 0.7