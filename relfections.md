# Reflections

## What went bad because of me?
-  spend too much time on hyperparameters tuning
    - spending time on hypeparametsrs is always a good thing but there much be a reason to go after it.
    - I spent time doing it mindlessly PRETENDING working on the problems
    - Became too lazy pushing back important stuff like trying different way of feeding image to model or even attemption to reduce noise

- Lack of discussion or meaningful analysis
    - not asking any question why model is not improving.
    - not questioning splitting cross validatiion strategy

- Sagittal T1 and T2 may look similar but they still have significant difference
    - most of the winner's solution seems to working separately on each plane not altogether like I did

- Getting too comfortable on already working code
    - Not use what was the reasons, but surely they are one of these
        - became too lazy to change code
        - Lack of motivation, excitement on the current problem
        - Overconfidence on the current position (Public Leaderboard was in silver medal range)   
        - Simply, unhealthy mind and body
            - This might cause making many bad decisions

- Ignore the fact that Local CV and LB is acting differently simply not sharing the same trend

## What could help me won if I had done?
- train each condition cooresponding series
    - What I did was include every series into one sample. Which did not help further improvement of the score in LB and PB. My observation is the model was leaning toward more to normal_mild severity to get more biase and became irrelevant from the real-world data.

- split with StratifiedGroupKFold targeting on only severity
    - seperating to 3 tables for each main conditions (considering left and right as the same)
    - What I did was split based on every conditions. Thus, the already unbalanced data is stil not intact. *SIMPLY OVERFITTING AND CAUSED BIG SHAKEDOWN.*

- resize image to specific size then crop with determined size
    - What I did was determined cropping size with ratio to image. It was consistent but not enough
    - I could have get a better view cropping region of interest because the area will always be the same.

- ignored the fact that label is noisy
    - What I did was only removed the label that was undefined or null.

## What I could have done better?
- I did not go after some ideas without answering why I did not
    - for example, using newly predicting coordinates instead of given one
        - noise in dataset is quite obvious but I did not see it
        - the problem is not it will work or not. It was I was not even bother to try it out

- SIMPLY, get healthy body and mind
    - more sleep
    - drink more water
    - eat slowly
    - eat at the same period of time
    - exercise more

- looking for more motivation others than winning an medal
    - opportunity for learning new thing
    - get out of comfort zone by trying new thing. challenge thing I don't know

- Could have tried bbox dectection on manual labeled

- Note taking
    - copy-paste experiement setup might be good for look up for changes that had been made but, it become harder and harder to backtracking
    - too less observation. Sometimes no observation of what happened after experiemnt completed
