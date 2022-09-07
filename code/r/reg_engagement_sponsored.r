data <- 'C:/Users/kas1112/Documents/research_social_media/data'
out <- paste0(data, '/out')
out_csv <- paste0(out, '/csv')

df_ig = read.csv(paste0(out_csv, '/engagement_regression_data.csv'))

model <- lm(engagement ~ sponsored + followers_num, data = df_ig)

model

sum(with(df_ig, sponsored == 0))

df_ig

model <- lm(engagement ~ num_caption_mentions + followers_num, data = df_ig)

model
