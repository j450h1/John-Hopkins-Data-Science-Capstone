#Load preprocessed ngram dataframes
#  df4 <- read.table('df4.txt', fill = TRUE, stringsAsFactors = FALSE) 
#  df3 <- read.table('df3.txt', fill = TRUE, stringsAsFactors = FALSE) 
#  df2 <- read.table('df2.txt', fill = TRUE, stringsAsFactors = FALSE)
#  input <- "how are you"

alternative_predictions<- function(input,df2,df3,df4){
  library(wordcloud)
  library(tm)
  library(sqldf)
  input <- tolower(input)
  output <- "" #Going to be a string of words"
  #Split into 3 words
  wordlist <- unlist(strsplit(input, split = ' '))
  first <- wordlist[1]; second <- wordlist[2]; third <- wordlist[3]  
  
  #See if the input has an exact match with either n-gram, using a backoff model
  #Start with 4gram down to bigram if necessary
  #If no bigram found, return "the" which is the most common result
  
  fourgram_query <- "SELECT * 
  FROM df4 
  WHERE V1 = '$first' AND V2 = '$second' AND V3 = '$third' "
  trigram_query <- "SELECT * 
  FROM df3 
  WHERE V1 = '$second' AND V2 = '$third' "
  bigram_query <- "SELECT * 
  FROM df2 
  WHERE V1 = '$third' "
  
  if (nrow(fn$sqldf(fourgram_query)) > 0){ #This means all three words found together
  # Find the 4gram with the maximum frequency out of this subset
  result <- fn$sqldf(fourgram_query)
  output <-  list(result$V4)
  test <- 4
  } else if (nrow(fn$sqldf(trigram_query)) > 0){ 
  # All three words are not found together so we look at second and third word
  result <- fn$sqldf(trigram_query)  
  output <-  list(result$V3)
  test <- 3
  } else if (nrow(fn$sqldf(bigram_query)) > 0){
  # Neither all three words or the preceding two words are found, so we only look for the word prior
  result <- fn$sqldf(bigram_query)  
  output <- list(result$V2)
  test <- 2
  } else {
  #predict the most common word "the" if its not in the model
  output <- list("the","is","thee","most","common","word","that","is","predicted")
  test <- 1
  }
  output <- paste(unlist(output), collapse = " ")
  wordcloud(output,colors=brewer.pal(6,"Dark2"),random.order=FALSE)    
  #print(output)
}  
#alternative_predictions(input,df2,df3,df4)
