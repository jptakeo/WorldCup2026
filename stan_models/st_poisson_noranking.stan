
data { int N; int T; array[N] int team_i, team_j, y_i, y_j; vector[N] game_weight; vector[T] prior_strength; }
parameters { vector[T] attack_raw, defense_raw; real eta; real<lower=0.01> sigma_att, sigma_def; real beta_prior; }
transformed parameters {
  vector[T] attack = attack_raw - mean(attack_raw);
  vector[T] defense = defense_raw - mean(defense_raw);
}
model {
  attack_raw ~ student_t(3, 0, sigma_att);
  defense_raw ~ student_t(3, 0, sigma_def);
  eta ~ normal(0, 1); beta_prior ~ normal(0, 1);
  sigma_att ~ cauchy(0, 2.5); sigma_def ~ cauchy(0, 2.5);
  for (n in 1:N) target += game_weight[n] * (poisson_log_lpmf(y_i[n] | attack[team_i[n]] - defense[team_j[n]] + eta) + poisson_log_lpmf(y_j[n] | attack[team_j[n]] - defense[team_i[n]] + eta));
}
