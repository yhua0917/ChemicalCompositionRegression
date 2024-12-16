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

## Future task; output target of DNN

Sensory evaluation of essential oils: While prediction of odor character may be of interest, there is currently no database available in sensory evaluation of essential oil. 
The availablity of sensory evaluation differs from the task on a task of single odorant, 
for which some databases are available to study the predictability of sensory evaluation. 
In this research, we did not forecast the odor type of essential oils using our architecture. 
If any database with odor type information is available, we will evaluate the predictability on our architecture.

The scientific name and/or varieties of the plant: The scientific name can be obtained via the web link of the entry "Detail" in the essential oil property table. 
However, they were unsuitable for learning and verification due to their diversity.
We attempted to compare plant varieties in the essential oil property table using the POWO database, 
but were unable to do so due to inconsistencies in the scientific names and lack of IDs.

