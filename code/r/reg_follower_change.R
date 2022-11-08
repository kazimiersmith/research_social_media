library(stargazer)

root <- 'C:/Users/kas1112/Documents/research_social_media'
data <- paste0(root, '/data')
out <- paste0(data, '/out')
reg_data <- paste0(out, '/estimation')
tab <- paste0(root, '/tab')

# Carrying capacity for follower growth model
carrying_cap <- 3500000

df_ig <- read.csv(paste0(reg_data, '/posts_panel.csv'))

# Terms in the follower growth regression
df_ig <- transform(df_ig,
  const_term = followers * (1 - followers / carrying_cap),
  spon_term = sponsored_posts * followers * (1 - followers / carrying_cap),
  posts_term = posts * followers * (1 - followers / carrying_cap),
  eng_term = engagement * followers * (1 - followers / carrying_cap)
)

write.csv(df_ig, paste0(reg_data, '/posts_panel_regression_test.csv'))

model1 <- lm(change_followers ~ const_term + spon_term + posts_term + eng_term, data = df_ig)
summary(model)

stargazer(model1, out = paste0(tab, '/results.tex'))