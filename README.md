## A simple DNN regression for the chemical composition in essential oil 

Although experimental design and methodological surveys for mono-molecular activity/property has been extensively investigated, 
those for chemical composition have received little attention, with the exception of a few prior studies. 
In this study, we configured three simple DNN regressors to predict essential oil property based on chemical composition. 
Despite showing overfitting due to the small size of dataset, all models were trained effectively in this study. 

## Essential oil database 

The property table for essential oils is collected from a website and reshaped.
The property table includes 'Essential Oil Name', 'Plant Name', 'Plant Tissue Name', 'Compound Name(s)' (chemical names only) and detailed web link. 
By the detailed web link in the property table, 
we can obtain the analytical table including 'Compound Name(s)' and 'Compound Percentage(%)'. 
The area% of gas chromatography (GC) is denoted as 'Compound Percentage(%)' in the analytical table. 
The molecular structure is able to obtain from 'Compound Name(s)'. 

[AromaDb]{https://bioinfo.cimap.res.in/aromadb/web_essential_oil.php}

