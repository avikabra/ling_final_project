library(readr)
library(ggplot2)
library(broom)

file_path <- file.path(getwd(), "fig-llama-subjectid-comb.csv")
df <- read_csv(file_path)

df[is.na(df)] <- 0
write_csv(df, "fig-gpt-comb.csv")

# Number of examples
nrow(df)

# Accuracies
num <- nrow(df[df$result==1, ])
den <- nrow(df)
print(paste("Accuracy:", round(num/den, 3)))

predictors <- c("obj", "vis", "soc", "cul")

for (predictor in predictors) {
  num <- nrow(df[df$result==1 & df[[predictor]]==1, ])
  den <- nrow(df[df[[predictor]]==1, ])
  print(paste(predictor, ":", round(num/den, 3)))
}

model <- glm(
  result ~ obj + vis + soc + cul, 
  data = df, 
  family = binomial
)

summary(model)
