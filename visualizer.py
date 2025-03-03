import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from plotly.subplots import make_subplots

# Read the JSON file
with open('community_software_dev.json', 'r') as f:
    tweets = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(tweets)

# Convert created_at to datetime
df['created_at'] = pd.to_datetime(df['created_at'])
df = df[df['created_at'] < '2025-03-03 06:00']

# Extract user metrics
df['followers_count'] = df['user'].apply(lambda x: x['followers_count'])
df['friends_count'] = df['user'].apply(lambda x: x['friends_count'])

# Convert string counts to integers
df['views_count'] = pd.to_numeric(df['views_count'])
df['quote_count'] = pd.to_numeric(df['quote_count'])
df['reply_count'] = pd.to_numeric(df['reply_count'])
df['retweet_count'] = pd.to_numeric(df['retweet_count'])
df['media_count'] = pd.to_numeric(df['media_count'])

# Define engagement metrics
engagement_metrics = ['views_count', 'quote_count', 'reply_count', 'retweet_count', 'media_count', 'followers_count', 'total_engagement']

# Calculate total engagement
df['total_engagement'] = df['views_count'] + df['quote_count'] + df['reply_count'] + df['retweet_count']

# Calculate new metrics
df['follower_following_ratio'] = df['followers_count'] / df['friends_count'].replace(0, 1)  # Avoid division by zero
df['tweet_length'] = df['full_text'].str.len()

# Calculate KPIs
avg_engagement = df['total_engagement'].mean()
median_engagement = df['total_engagement'].median()
engagement_std = df['total_engagement'].std()

# Calculate probability metrics
total_tweets = len(df)
above_avg_tweets = len(df[df['total_engagement'] > avg_engagement])
above_median_tweets = len(df[df['total_engagement'] > median_engagement])
short_tweets = len(df[df['tweet_length'] < 150])
high_engagement_short_tweets = len(df[(df['tweet_length'] < 150) & (df['total_engagement'] > avg_engagement)])

# Calculate percentages
above_avg_prob = (above_avg_tweets / total_tweets) * 100
above_median_prob = (above_median_tweets / total_tweets) * 100
short_tweet_success_rate = (high_engagement_short_tweets / short_tweets) * 100

# Calculate date range for the filtered data
start_date = df['created_at'].min().strftime('%B %d, %Y')
end_date = df['created_at'].max().strftime('%B %d, %Y')

# Create KPI text
kpi_text = f"""
Key Performance Indicators:
• Average tweet engagement: {avg_engagement:.1f} (median: {median_engagement:.1f})
• {above_avg_prob:.1f}% of tweets exceed the average engagement of {avg_engagement:.1f}
• {above_median_prob:.1f}% of tweets exceed the median engagement of {median_engagement:.1f}
• Short tweets (<150 chars) have a {short_tweet_success_rate:.1f}% chance of exceeding average engagement
• Most engaging tweets have {df['media_count'].mode()[0]} media items
• Top 10% of tweets average {df['total_engagement'].quantile(0.9):.1f} engagements
• Tweets with >1000 followers are {len(df[df['followers_count'] > 1000])/total_tweets*100:.1f}% of the dataset
• Engagement standard deviation: {engagement_std:.1f}
"""

# Create follower count ranges and calculate probabilities
follower_ranges = [0, 50, 100, 500, 1000, float('inf')]
follower_labels = ['0-50', '51-100', '101-500', '501-1000', '1000+']
engagement_probabilities = []

for i in range(len(follower_ranges)-1):
    mask = (df['followers_count'] > follower_ranges[i]) & (df['followers_count'] <= follower_ranges[i+1])
    group_tweets = df[mask]
    if len(group_tweets) > 0:
        prob = (len(group_tweets[group_tweets['total_engagement'] > avg_engagement]) / len(group_tweets)) * 100
        engagement_probabilities.append(prob)
    else:
        engagement_probabilities.append(0)

# Create a figure with subplots (3x2 layout)
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Media Count vs Total Engagement',
                   'User Followers vs Total Engagement',
                   'Engagement Distribution by Media Count',
                   'Follower/Following Ratio vs Engagement',
                   'Tweet Length vs Engagement',
                   'Engagement Probability by Follower Count')
)

# 1. Scatter plot: Media count vs Total Engagement
fig.add_trace(
    go.Scatter(x=df['media_count'], y=df['total_engagement'],
               mode='markers', name='Media vs Engagement',
               marker=dict(size=8, opacity=0.6)),
    row=1, col=1
)

# Add trendline for media count
z = np.polyfit(df['media_count'], df['total_engagement'], 1)
p = np.poly1d(z)
fig.add_trace(
    go.Scatter(x=df['media_count'], y=p(df['media_count']),
               mode='lines', name='Trendline',
               line=dict(color='red', width=2)),
    row=1, col=1
)

# 2. Scatter plot: User Followers vs Total Engagement with 2D Histogram overlay
fig.add_trace(
    go.Histogram2d(
        x=df['followers_count'],
        y=df['total_engagement'],
        colorscale='Viridis',
        nbinsx=30,
        nbinsy=30,
        zauto=False,
        zmax=10,
        opacity=0.7,
        hoverinfo='skip'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=df['followers_count'], y=df['total_engagement'],
               mode='markers', name='Followers vs Engagement',
               marker=dict(size=8, opacity=0.6, color='white', line=dict(width=1))),
    row=1, col=2
)

# Add trendline for followers
z = np.polyfit(df['followers_count'], df['total_engagement'], 1)
p = np.poly1d(z)
fig.add_trace(
    go.Scatter(x=df['followers_count'], y=p(df['followers_count']),
               mode='lines', name='Trendline',
               line=dict(color='red', width=2)),
    row=1, col=2
)

# 3. Box plot: Engagement by Media Count
fig.add_trace(
    go.Box(x=df['media_count'], y=df['total_engagement'],
           name='Engagement by Media',
           boxpoints='outliers'),
    row=2, col=1
)

# 4. Scatter plot: Follower/Following Ratio vs Engagement with 2D Histogram
fig.add_trace(
    go.Histogram2d(
        x=df['follower_following_ratio'],
        y=df['total_engagement'],
        colorscale='Viridis',
        nbinsx=30,
        nbinsy=30,
        zauto=False,
        zmax=10,
        opacity=0.7,
        hoverinfo='skip'
    ),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(x=df['follower_following_ratio'], y=df['total_engagement'],
               mode='markers', name='Ratio vs Engagement',
               marker=dict(size=8, opacity=0.6, color='white', line=dict(width=1))),
    row=2, col=2
)

# Add trendline for follower/following ratio
z = np.polyfit(df['follower_following_ratio'], df['total_engagement'], 1)
p = np.poly1d(z)
fig.add_trace(
    go.Scatter(x=df['follower_following_ratio'], y=p(df['follower_following_ratio']),
               mode='lines', name='Trendline',
               line=dict(color='red', width=2)),
    row=2, col=2
)

# 5. Scatter plot: Tweet Length vs Engagement with 2D Histogram
fig.add_trace(
    go.Histogram2d(
        x=df['tweet_length'],
        y=df['total_engagement'],
        colorscale='Viridis',
        nbinsx=30,
        nbinsy=30,
        zauto=False,
        zmax=10,
        opacity=0.7,
        hoverinfo='skip'
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=df['tweet_length'], y=df['total_engagement'],
               mode='markers', name='Length vs Engagement',
               marker=dict(size=8, opacity=0.6, color='white', line=dict(width=1))),
    row=3, col=1
)

# Add trendline for tweet length
z = np.polyfit(df['tweet_length'], df['total_engagement'], 1)
p = np.poly1d(z)
fig.add_trace(
    go.Scatter(x=df['tweet_length'], y=p(df['tweet_length']),
               mode='lines', name='Trendline',
               line=dict(color='red', width=2)),
    row=3, col=1
)

# 6. Bar chart: Engagement probability by follower count
fig.add_trace(
    go.Bar(
        x=follower_labels,
        y=engagement_probabilities,
        marker_color='#000000',
        text=[f'{prob:.1f}%' for prob in engagement_probabilities],
        textposition='auto',
        name='Engagement Probability'
    ),
    row=3, col=2
)

# Update layout
fig.update_layout(
    height=1500,
    width=1200,
    title_text="Tweet Engagement Analysis",
    showlegend=False,
    grid=dict(rows=3, columns=2, pattern="independent")
)

# Update axes labels
fig.update_xaxes(title_text="Number of Media Items", row=1, col=1)
fig.update_xaxes(title_text="Number of Followers", row=1, col=2)
fig.update_xaxes(title_text="Number of Media Items", row=2, col=1)
fig.update_xaxes(title_text="Follower/Following Ratio", row=2, col=2)
fig.update_xaxes(title_text="Tweet Length (characters)", row=3, col=1)
fig.update_xaxes(title_text="Follower Count Range", row=3, col=2)

fig.update_yaxes(title_text="Total Engagement", row=1, col=1)
fig.update_yaxes(title_text="Total Engagement", row=1, col=2)
fig.update_yaxes(title_text="Total Engagement", row=2, col=1)
fig.update_yaxes(title_text="Total Engagement", row=2, col=2)
fig.update_yaxes(title_text="Total Engagement", row=3, col=1)
fig.update_yaxes(title_text="Probability of Above Average Engagement (%)", row=3, col=2)

# Save the combined plot
fig.write_html("tweet_analysis.html")

# Create index.html with KPIs and styling
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X (Twitter) Engagement Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f8fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #000000;
            text-align: center;
            margin-bottom: 30px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 4px solid #000000;
        }}
        .kpi-card h3 {{
            color: #000000;
            margin-top: 0;
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: bold;
            color: #14171A;
            margin: 10px 0;
        }}
        .kpi-description {{
            color: #657786;
            font-size: 14px;
        }}
        .visualization {{
            margin-top: 40px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        iframe {{
            width: 100%;
            height: 1500px;
            border: none;
            overflow: hidden;
            scrollbar-width: none;  /* Firefox */
            -ms-overflow-style: none;  /* IE and Edge */
        }}
        iframe::-webkit-scrollbar {{
            display: none;  /* Chrome, Safari, Opera */
        }}
        .explanation {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
            border-left: 4px solid #000000;
        }}
        .explanation h2 {{
            color: #000000;
            margin-top: 0;
        }}
        .explanation p {{
            color: #657786;
            margin-bottom: 15px;
        }}
        .explanation ul {{
            color: #657786;
            margin: 0;
            padding-left: 20px;
        }}
        .explanation li {{
            margin-bottom: 8px;
        }}
        .data-source {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
            border-left: 4px solid #000000;
        }}
        .data-source h2 {{
            color: #000000;
            margin-top: 0;
        }}
        .data-source p {{
            color: #657786;
            margin-bottom: 15px;
        }}
        .data-source a {{
            color: #000000;
            text-decoration: none;
            border-bottom: 1px solid #000000;
        }}
        .data-source a:hover {{
            border-bottom: 2px solid #000000;
        }}
        .data-source ul {{
            color: #657786;
            margin: 0;
            padding-left: 20px;
        }}
        .data-source li {{
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>X Engagement Analysis</h1>
        
        <div class="data-source">
            <h2>Data Source</h2>
            <p>This analysis is based on posts from the "Software Development" community on X, collected from {start_date} to {end_date}.</p>
            <ul>
                <li>Raw data: <a href="community_software_dev.json">community_software_dev.json</a></li>
                <li>Analysis script: <a href="visualizer.py">visualizer.py</a></li>
            </ul>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <h3>Average Engagement</h3>
                <div class="kpi-value">{avg_engagement:.1f}</div>
                <div class="kpi-description">Median: {median_engagement:.1f}</div>
            </div>
            
            <div class="kpi-card">
                <h3>Above Average Probability</h3>
                <div class="kpi-value">{above_avg_prob:.1f}%</div>
                <div class="kpi-description">Posts exceeding {avg_engagement:.1f} engagements</div>
            </div>
            
            <div class="kpi-card">
                <h3>Short Post Success Rate</h3>
                <div class="kpi-value">{short_tweet_success_rate:.1f}%</div>
                <div class="kpi-description">Posts under 150 characters exceeding average</div>
            </div>
            
            <div class="kpi-card">
                <h3>Top 10% Threshold</h3>
                <div class="kpi-value">{df['total_engagement'].quantile(0.9):.1f}</div>
                <div class="kpi-description">Engagements for top performing posts</div>
            </div>
            
            <div class="kpi-card">
                <h3>Most Common Media Count</h3>
                <div class="kpi-value">{df['media_count'].mode()[0]}</div>
                <div class="kpi-description">Media items in engaging posts</div>
            </div>
            
            <div class="kpi-card">
                <h3>Large Accounts</h3>
                <div class="kpi-value">{len(df[df['followers_count'] > 1000])/total_tweets*100:.1f}%</div>
                <div class="kpi-description">Posts from accounts with >1000 followers</div>
            </div>
        </div>

        <div class="explanation">
            <h2>Understanding Engagement</h2>
            <p>Engagement is calculated as the sum of all interactions with a post:</p>
            <ul>
                <li>Views: Number of times the post was viewed</li>
                <li>Quotes: Number of times the post was quoted</li>
                <li>Replies: Number of direct replies to the post</li>
                <li>Reposts: Number of times the post was reposted</li>
            </ul>
            <p>For example, if a post has 100 views, 5 quotes, 3 replies, and 2 reposts, its total engagement would be 110.</p>
            <p>This metric helps understand the overall reach and interaction level of posts, with higher numbers indicating more successful content.</p>
        </div>

        <div class="visualization">
            <iframe src="tweet_analysis.html"></iframe>
        </div>
    </div>
</body>
</html>
"""

# Save the index.html file
with open('index.html', 'w') as f:
    f.write(html_content)

# Print some basic statistics
print("\nBasic Statistics:")
print(df[engagement_metrics].describe())

print("\nAdditional Metrics Statistics:")
print(df[['follower_following_ratio', 'tweet_length']].describe())

print("\nCorrelation with Total Engagement:")
new_metrics = ['follower_following_ratio', 'tweet_length']
all_metrics = engagement_metrics + new_metrics
correlations = df[all_metrics].corr()['total_engagement'].sort_values(ascending=False)
print(correlations)
