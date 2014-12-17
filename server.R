library(shiny)
library(shinyIncubator)
#Load preprocessed ngram dataframes
df4 <- read.table('df4.txt', fill = TRUE, stringsAsFactors = FALSE) 
df3 <- read.table('df3.txt', fill = TRUE, stringsAsFactors = FALSE) 
df2 <- read.table('df2.txt', fill = TRUE, stringsAsFactors = FALSE)
source("ngram_prediction_model.R")
source("alternative_predictions_wordcloud.R")

shinyServer(function(input, output, session) {
  numWords <- reactive({
    paste(length(unlist(strsplit(input$text," "))),'WORDS ENTERED.\n\nAfter entering exactly 3 words, Click PREDICT! to see the 4th word.\n\nEXAMPLE: Type "What are you" and you\'ll get a prediction of "doing".')
  })  
  output$loading <- renderText({
    if(input$Button!=0) return("MESSAGE: Try another three word phrase!")
    withProgress(session, min=1, max=30, expr={
      for(i in 1:30) {
        setProgress(message = 'Loading...',
                    detail = 'This should take a few seconds...',
                    value=i)
        print(i)
        Sys.sleep(0.1)
      }
      #df4 <- read.table('df4.txt', fill = TRUE, stringsAsFactors = FALSE) 
      #df3 <- read.table('df3.txt', fill = TRUE, stringsAsFactors = FALSE) 
      #df2 <- read.table('df2.txt', fill = TRUE, stringsAsFactors = FALSE)
      #source("ngram_prediction_model.R")
      "MESSAGE: Done Loading!"
    })
  })
  output$value <- renderText({numWords()
  })
  output$text1 <- renderText({
    #Requires predict button to be pressed before a prediction is made.
    if(input$Button==0) return(NULL)
    #withProgress(message = 'Calculating the most likely 4th word...', value = 0.1, {
    isolate({
      #predict_word(input$text,df2,df3,df4)
      withProgress(session, min=1, max=30, expr={
        for(i in 1:30) {
          setProgress(message = 'Calculating the most likely 4th word...',
                    detail = 'This should take a few seconds...',
                    value=i)
          print(i)
          Sys.sleep(0.1)
        }
        predict_word(input$text,df2,df3,df4)
      })
    })
  })
  output$plot <- renderPlot({
    if(input$Button==0) return(NULL)
    isolate({
      withProgress(session, min=1, max=30, expr={
        for(i in 1:30) {
          setProgress(message = 'Generating word cloud...',
                      detail = 'This should take a few seconds...',
                      value=i)
          print(i)
          Sys.sleep(0.1)
        }
        tryCatch({
          alternative_predictions(input$text,df2,df3,df4);
        }, interrupt = function(ex) {
          cat("No alternatives found. Try another phrase.");
          print(ex);
        }, error = function(ex) {
          plot(0,0,type='n',axes=FALSE,ann=FALSE);#empty plot
          text(0,0,cex=1.75,"No alternative predictions. Try another phrase.")                  
        }, finally = {
          #cat("Releasing resources...");
        }) # tryCatch()
      })
    })
  })
})

