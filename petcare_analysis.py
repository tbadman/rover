"""
Main analysis script for excersises in 'instructions.pdf' for submission to Rover.com.

I chose to work on excersises II and III since I feel they have the most
direct impact on the pets themselves, and excersise VI to showcase some
(albeit very minor) ML (regression) techniques.

Note: excersise I is covered as a unit-test module in 'test_connections.py'

Requirements:
Python 3.x
requirements.txt dependencies

Run:
python petcare_analysis.py

December 2018
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy import stats

import config
from db_interact import DBInteract

def conversations_and_bookings():
    """EXCERSISE II: Conversation/Booking/Review study"""

    db_conn = DBInteract(config.DATABASE)

    reviewers_query = """
        SELECT cr.reviewer_id,
                cr.stars,
                CASE 
                    WHEN ss.provider_id=cr.reviewer_id THEN 'provider'
                    ELSE 'owner'
                END as reviewer
        FROM conversations_review cr
        JOIN conversations_conversation cc on (cc.id = cr.conversation_id)
        JOIN services_service ss on (ss.id = cc.service_id)
        """

    services_query = """
        SELECT cc.requester_id as owner, 
               ss.provider_id as provider
        FROM conversations_conversation cc
        JOIN services_service ss  on (ss.id = cc.service_id)
        WHERE cc.booked_at is not null 
        AND cc.cancelled_at is null
        AND cc.start_date < date('now')
        """

    reviewers = pd.read_sql_query(reviewers_query, db_conn.conn)
    services = pd.read_sql_query(services_query, db_conn.conn)

    # number of reviews per 'reviewer type' (calculated in query)
    # reviewer type can be provider or owner
    n_reviews = reviewers.groupby(['reviewer']).reviewer_id.count()
    n_total_services = len(services.index)

    # number of reviews per distinct 'reviewer type'
    distinct_reviews = reviewers.groupby(['reviewer']).reviewer_id.nunique()
    distinct_total = services.nunique()

    p_owner_reviews = round(distinct_reviews.iloc[0]/distinct_total.iloc[0]*100, 3)
    p_provider_reviews = round(distinct_reviews.iloc[1]/distinct_total.iloc[1]*100, 3)
    print("Number of distinct owners: {}".format(distinct_total.iloc[0]))
    print("Number of distinct providers: {}".format(distinct_total.iloc[1]))
    print("Number of distinct owners with reviews: {}".format(distinct_reviews.iloc[0]))
    print("Number of distinct providers with reviews: {}".format(distinct_reviews.iloc[1])) 
    print('% of owners who leave reviews: {}'.format(p_owner_reviews))
    print('% of providers who leave reviews: {}'.format(p_provider_reviews))

    mean_reviews = reviewers.groupby(['reviewer']).stars.mean()
    std_reviews = reviewers.groupby(['reviewer']).stars.std()

    # explored using a scatter plot (below), decided a normpdf would be
    # more clear to the reader because it emphasizes the differences in
    # distributions of reviews between the two categories.
    # ....
    # mean = [mean_reviews.iloc[i] for i in range(len(mean_reviews.index))]
    # std = [std_reviews.iloc[i] for i in range(len(mean_reviews.index))]
    # categories = [i for i in range(2)]
    # plt.errorbar(categories, mean, std, 0, fmt='o', capsize=2)

    # plot normal distribution of star ranges for both reviewer types
    print("Average review for pet owners: {}".format(mean_reviews.iloc[0]))
    print("Average review for providers: {}".format(mean_reviews.iloc[1]))
    star_range = np.linspace(0, 5, 100)
    plt.plot(star_range,
             mlab.normpdf(star_range, mean_reviews.iloc[0], std_reviews.iloc[0]),
             label='owner')
    plt.plot(star_range,
             mlab.normpdf(star_range, mean_reviews.iloc[1], std_reviews.iloc[1]),
             label='provider')

    # plot formatting
    plt.legend()
    plt.title('Probability Distribution of Reviews')
    plt.xlabel('# Stars')

    plt.grid()
    plt.show()

def recent_daily_booking_rate():
    """EXCERSISE III: Daily booking rate study"""

    db_conn = DBInteract(config.DATABASE)

    booked_query = """
        SELECT strftime('%m/%d',t_conversations.added) as added, 
               n_conversations, 
               coalesce(n_booked,0) as n_booked,
               cast(coalesce(n_booked,0) as float)/n_conversations as fraction_booked
        FROM (SELECT date(added) as added, count(*) as n_conversations
                FROM conversations_conversation cc
                GROUP BY date(added)) t_conversations
        LEFT JOIN (SELECT date(added) as added, count(*) as n_booked
                        FROM conversations_conversation cc
                        WHERE booked_at is not null and cancelled_at is null
                        GROUP BY date(added)) t_booked on (t_conversations.added = t_booked.added)
        WHERE t_conversations.added >= date('2018-08-02', '-90 day')
        ORDER BY t_conversations.added
        """

    time_to_book_query = """
        SELECT date(added) as added, 
               round(julianday(booked_at)-julianday(added),6) as time_to_book
        FROM conversations_conversation cc
        WHERE added >= date('2018-08-02', '-90 day')
        """

    booked_per_day = pd.read_sql_query(booked_query, db_conn.conn)
    time_to_book = pd.read_sql_query(time_to_book_query, db_conn.conn)

    # PART 1: verify downward trend
    plt.plot(booked_per_day.added, booked_per_day.fraction_booked)

    # vertical line at beginning of downward trend
    plt.plot(['07/28' for i in range(10)], [i/10 for i in range(0, 10)], 'k--')

    # formatting
    ticks = [date for idx, date in enumerate(booked_per_day.added) if idx%5 == 0]
    plt.xticks(ticks, rotation='45')
    plt.xlabel('Date (MM/DD)')
    plt.title('Daily Booking Rate')

    plt.grid()
    plt.show()

    # PART 2: avg time to book
    # My thinking is that the drop-off in booking rate is due to some
    # average amount of time between 'conversation start' and 'booking start'.
    time_to_book = time_to_book.dropna() # clean time_to_book DF for fitting

    # plot time-to-book histogram (value is calculated in query)
    n, bins, patches = plt.hist(time_to_book.time_to_book, bins=100, density=1, alpha=0.2)

    # fit a skew normal PDF to 'time_to_book' (helps visualize mean time-to-book)
    ae, loce, scalee = stats.skewnorm.fit(time_to_book.time_to_book)
    p = stats.skewnorm.pdf(bins, ae, loce, scalee)
    plt.plot(bins, p, 'r--', linewidth=2)

    #formatting
    plt.xlabel('Time to Book (days)')
    plt.title('Time Between Conversation Start and Booking')

    plt.grid()
    plt.show()

def search_engine_marketing():
    """EXCERSISE VI: Determine value of search engine marketing change."""

    db_conn = DBInteract(config.DATABASE)

    new_goog_users_query = """
        SELECT date(date_joined) as date_joined, 
               count(*) as n_new_users,
               julianday(date_joined) as julian_date_joined
        FROM people_person
        WHERE channel='Google'
        GROUP BY date(date_joined)
        ORDER BY date(date_joined)
        """

    new_goog_users = pd.read_sql_query(new_goog_users_query, db_conn.conn)

    fig, ax = plt.subplots()

    # julian date equivalent of Google marketing change on 2018-05-04
    julian_date_cutoff = new_goog_users.loc[new_goog_users['date_joined'] == '2018-05-04', 'julian_date_joined'].iloc[0]

    plt.plot(new_goog_users.julian_date_joined, new_goog_users.n_new_users, '.')
    plt.plot([julian_date_cutoff for i in range(100)], [i for i in range(100)], 'k--')

    # create sub-dataframes for before and after change on 2018-05-04
    pre_df = new_goog_users.loc[new_goog_users['julian_date_joined'] < julian_date_cutoff, :]
    post_df = new_goog_users.loc[new_goog_users['julian_date_joined'] >= julian_date_cutoff, :]

    # create training set and test set from pre_df
    x_train = pre_df.julian_date_joined.as_matrix().reshape(len(pre_df.index), 1)[:int(len(pre_df.index)/2.)]
    y_train = pre_df.n_new_users.as_matrix().reshape(len(pre_df.index), 1)[:int(len(pre_df.index)/2.)]
    x_test = pre_df.julian_date_joined.as_matrix().reshape(len(pre_df.index), 1)
    y_test = pre_df.n_new_users.as_matrix().reshape(len(pre_df.index), 1)

    # Create linear regression model and fit training set
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    # test model for fit accuracy (not useful for report but good sanity check)
    y_pred = regr.predict(x_test)
    error = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # plot (I'm just plotting the model over the full train/test range for visualization)
    plt.plot(x_train, regr.predict(x_train), 'b', linewidth=3)
    plt.plot(x_test, regr.predict(x_test), 'b', linewidth=3)

    # use model to predict new_users w/o change and plot
    x_post = post_df.julian_date_joined.as_matrix().reshape(len(post_df.index), 1)
    y_post = regr.predict(x_post)
    plt.plot(post_df.julian_date_joined, y_post, 'b--', linewidth=3)

    # since julian_time is used for regression model, I had to relabel x-axis for readability
    labels = new_goog_users.date_joined.iloc[::int(len(new_goog_users.index)/8)]
    ax.set_xticklabels(labels, rotation=45)

    # formatting/plotting
    plt.title('New Users Acquired Through Google')
    plt.xlabel('Date')
    plt.grid()
    plt.show()

    # algebra!
    est_new_users_wo_change = int(sum(y_post)[0])
    new_users_w_change = sum(post_df.n_new_users)

    est_cost_wo_change = 30. * est_new_users_wo_change # $30 per account pre-change
    cost_w_change = 207180. # given in instructions.pdf

    cost_pp_w_change = cost_w_change/new_users_w_change

    additional_acc_created = new_users_w_change - est_new_users_wo_change
    additional_cost_created = cost_w_change - est_cost_wo_change

    marginal_cost_pp = round(additional_cost_created/additional_acc_created, 2)

    print('Estimated new users without change: {}'.format(est_new_users_wo_change))
    print('Estimated marketing cost without change: {}'.format(est_cost_wo_change))
    print('Number of users acquired through Google (post change): {}'.format(new_users_w_change))
    print('Average cost per account (post change): ${}'.format(cost_pp_w_change))
    print('Marginal cost per additional account from change: ${}'.format(marginal_cost_pp))

if __name__ == "__main__":

    conversations_and_bookings()
    recent_daily_booking_rate()
    search_engine_marketing()
