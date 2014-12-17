library(shiny)
library(shinyIncubator)

shinyUI(fluidPage(
  # Sidebar
  progressInit(),
  sidebarPanel(width = 11,
    textInput("text", label = h2("Next Text"),value = "Loading instructions, please wait:"),           
    actionButton("Button", "PREDICT!"),
    progressInit(),
    fluidRow(column(11, verbatimTextOutput("value"))),
    fluidRow(column(11, verbatimTextOutput("loading")))
  ),
  mainPanel(
    hr(),
    h4("The predicted word is:"),
    hr(),
    textOutput("text1"),
    hr(),
    h4("Some other possibilities:"),
    hr(),
    plotOutput("plot"),
    hr()
  )
))