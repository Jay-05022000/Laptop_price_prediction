Task:
You have to predict the price of given laptop based on given specifications such as Inches,Ram,Weight,Op_sys,
memory,gpu,Screen_Resolution,company,Type_Name	& cpu.

Along with above task I am testing usefullness of correlation matrics to perform feature elimination.


Model Versions:

1) Version 1: Random forest regression witout feature elemination
         (a): with introduction of feature elemination (those which have low correlation with label)
         (b): Feature elimination (eliminating 'gpu' feature)
2) Version 2: SVR with scaled dataset and without feature elemination


Results:

Version 1 :mean absolute error is: 136.38088888888888
           mean absolute percentage error is: 0.1445928277205149
           R-Squares score is: 0.8899338480010867
           Adjusted R-Squared score is: 0.8874035916332955

Version 1(a) : mean absolute error is: 159.38968888888888
               mean absolute percentage error is: 0.16436154712327572
               R-Squares score is: 0.8636026553318473
               Adjusted R-Squared score is: 0.8626856983929018

Version 1(b) : mean absolute error is: 144.02326666666667
               mean absolute percentage error is: 0.13978548704948562
               R-Squares score is: 0.8481887261550035
               Adjusted R-Squared score is: 0.8470396080182457

Version 2 : mean absolute error is: 193.19833333333335
            mean absolute percentage error is: 0.20314361080660118
            R-Squares score is: 0.7577458788129114
            Adjusted R-Squared score is: 0.7521768185557369


Conclusion:

When performing feature elimination with (1(a)) ,R-squared & adjusted R-Square is good in comparison to 1(b) but
error increases in comparison 1(b).

For close prediction to true value,removing features that have high correlation with each other but low with label
is a good option,although R parameters decreases.