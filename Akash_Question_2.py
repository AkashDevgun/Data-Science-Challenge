import argparse
import numpy as np 
from csv import DictReader, DictWriter
import glob
import os



def End ():

    print "Question 2 Complete"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Mining HW2')
    
    args = parser.parse_args()

    allfilenames = glob.glob("*.TXT")
    with open('outputfile.TXT', 'w') as outfile:
         for fname in allfilenames:
             with open(fname) as infile:
                 for line in infile:
                     outfile.write(line)

    femaledictionary = {}
    maledictionary = {} 
    combinedictinary = {} 
    yearlist = []  
    with open("outputfile.TXT", "r") as alllines:
        array = []
        for line in alllines:
            fields = line.split(',')
            tupppp = []
            val = int(fields[4])
            yearval = int(fields[2])
            yearlist.append(yearval)
            if fields[1] == 'F':
                if femaledictionary.has_key(fields[3]):                
                    femaledictionary[fields[3]] = femaledictionary.get(fields[3]) + val
                else:
                    femaledictionary.update({fields[3]:val})
            else:
                if maledictionary.has_key(fields[3]):
                    maledictionary[fields[3]] = maledictionary.get(fields[3]) + val
                else:
                    maledictionary.update({fields[3]:val})

            if combinedictinary.has_key(fields[3]):
                combinedictinary[fields[3]] = combinedictinary.get(fields[3]) + val
            else:
                combinedictinary.update({fields[3]:val})

    femalefinaldict = sorted(femaledictionary.items(), key=lambda x: x[1], reverse = True)
    malefinaldict = sorted(maledictionary.items(), key=lambda x: x[1], reverse = True)
    combinedinaldict = sorted(combinedictinary.items(), key=lambda x: x[1], reverse = True)
    print "Most Popular Female Name"
    print femalefinaldict[0]
    print "Most Popular Male Name"
    print malefinaldict[0]
    print "Most Popular Name of Either Gender"
    print combinedinaldict[0]

    Maxyear = max(yearlist)
    Minyear = min(yearlist)



    femaledictionary45 = {}
    maledictionary45 = {}    
    with open("outputfile.TXT", "r") as alllines:
        array = []
        for line in alllines:
            fields = line.split(',')
            tupppp = []
            val = int(fields[4])
            year = int(fields[2])
            if fields[1] == 'F' and year == 1945:
                if femaledictionary45.has_key(fields[3]):
                    femaledictionary45[fields[3]] = femaledictionary45.get(fields[3]) + val
                else:
                    femaledictionary45.update({fields[3]:val})
            else:
                if fields[1] == 'M' and year == 1945:
                    if maledictionary45.has_key(fields[3]):
                        maledictionary45[fields[3]] = maledictionary45.get(fields[3]) + val
                    else:
                        maledictionary45.update({fields[3]:val})

    commondict1945 = {}                    
    for k,v in femaledictionary45.iteritems():
        if (k in maledictionary45):                    
            min_val = min(femaledictionary45[k], maledictionary45[k])
            commondict1945.update({k:min_val})
    finalcommondict1945 = sorted(commondict1945.items(), key=lambda x: x[1], reverse = True)
    print "Most ambiguous name in 1945"
    print finalcommondict1945[0]

    



    femaledictionary13 = {}
    maledictionary13 = {}    
    with open("outputfile.TXT", "r") as alllines:
        array = []
        for line in alllines:
            fields = line.split(',')
            tupppp = []
            val = int(fields[4])
            year = int(fields[2])
            if fields[1] == 'F' and year == 2013:
                if femaledictionary13.has_key(fields[3]):
                    femaledictionary13[fields[3]]  = femaledictionary13.get(fields[3]) + val
                else:
                    femaledictionary13.update({fields[3]:val})
            else:
                if fields[1] == 'M' and year == 2013:
                    if maledictionary13.has_key(fields[3]):                        
                        maledictionary13[fields[3]] = maledictionary13.get(fields[3]) + val
                    else:
                        maledictionary13.update({fields[3]:val})

    commondict2013 = {}

    for k,v in femaledictionary13.iteritems():
        if (k in maledictionary13): 
            min_val = min(femaledictionary13[k], maledictionary13[k])
            commondict2013.update({k:min_val})
    finalcommondict2013 = sorted(commondict2013.items(), key=lambda x: x[1], reverse = True)
    print "Most ambiguous name in 2013"
    print finalcommondict2013[0]
    
    



    femaledictionarybefore1980 = {}
    femaledictionaryafter1980 = {}    
    with open("outputfile.TXT", "r") as alllines:
        array = []
        for line in alllines:
            fields = line.split(',')
            tupppp = []
            val = int(fields[4])
            year = int(fields[2])
            if fields[1] == 'F' and year < 1980:
                if femaledictionarybefore1980.has_key(fields[3]):
                    femaledictionarybefore1980[fields[3]] = femaledictionarybefore1980.get(fields[3]) + val
                else:
                    femaledictionarybefore1980.update({fields[3]:val})
            else:
                if fields[1] == 'F' and year >= 1980:
                    if femaledictionaryafter1980.has_key(fields[3]):
                        femaledictionaryafter1980[fields[3]] = femaledictionaryafter1980.get(fields[3]) + val
                    else:
                        femaledictionaryafter1980.update({fields[3]:val})

    female_percentage = {}
    female_change = {}
    for k,v in femaledictionarybefore1980.iteritems():
        if (k in femaledictionaryafter1980): 
            percent = float(femaledictionaryafter1980[k] - femaledictionarybefore1980[k])/femaledictionarybefore1980[k]
            change = femaledictionaryafter1980[k] - femaledictionarybefore1980[k]
            female_percentage.update({k:percent})
            female_change.update({k:change})

    finalfemalepercentage = sorted(female_percentage.items(), key=lambda x: x[1], reverse = True)
    print "Large percentage increase in Female Name"
    print finalfemalepercentage[0]
    finalfemalepercentage = sorted(female_percentage.items(), key=lambda x: x[1])
    print "Large percentage decrease in Female Name"
    print finalfemalepercentage[0]

    finalfemalechange = sorted(female_change.items(), key=lambda x: x[1], reverse = True)
    print "Large increase in Female Name"
    print finalfemalechange[0]
    finalfemalechange = sorted(female_change.items(), key=lambda x: x[1])
    print "Large percentage in Female Name"
    print finalfemalechange[0]









    maledictionarybefore1980 = {}
    maledictionaryafter1980 = {}    
    with open("outputfile.TXT", "r") as alllines:
        array = []
        for line in alllines:
            fields = line.split(',')
            tupppp = []
            val = int(fields[4])
            year = int(fields[2])
            if fields[1] == 'M' and year < 1980:
                if maledictionarybefore1980.has_key(fields[3]):
                    maledictionarybefore1980[fields[3]] = maledictionarybefore1980.get(fields[3]) + val
                else:
                    maledictionarybefore1980.update({fields[3]:val})
            else:
                if fields[1] == 'M' and year >= 1980:
                    if maledictionaryafter1980.has_key(fields[3]):
                        maledictionaryafter1980[fields[3]] = maledictionaryafter1980.get(fields[3]) + val
                    else:
                        maledictionaryafter1980.update({fields[3]:val})

    male_percentage = {}
    male_change = {}
    for k,v in maledictionarybefore1980.iteritems():
        if (k in maledictionaryafter1980): 
            percent = float(maledictionaryafter1980[k] - maledictionarybefore1980[k])/maledictionarybefore1980[k]
            change = maledictionaryafter1980[k] - maledictionarybefore1980[k]
            male_percentage.update({k:percent})
            male_change.update({k:change})

    finalmalepercentage = sorted(male_percentage.items(), key=lambda x: x[1], reverse = True)
    print "Large percentage increase in Male Name"
    print finalmalepercentage[0]
    finalmalepercentage = sorted(male_percentage.items(), key=lambda x: x[1])
    print "Large percentage decrease in Male Name"
    print finalmalepercentage[0]


    finalmalepechange = sorted(male_change.items(), key=lambda x: x[1], reverse = True)
    print "Large increase in Male Name"
    print finalmalepechange[0]
    finalmalepechange = sorted(male_change.items(), key=lambda x: x[1])
    print "Large decrease in Male Name"
    print finalmalepechange[0]
    


    combineddictionarybefore1980 = {}
    combineddictionaryafter1980 = {}    
    with open("outputfile.TXT", "r") as alllines:
        array = []
        for line in alllines:
            fields = line.split(',')
            tupppp = []
            val = int(fields[4])
            year = int(fields[2])
            if year < 1980:
                if combineddictionarybefore1980.has_key(fields[3]):
                    combineddictionarybefore1980[fields[3]] = combineddictionarybefore1980.get(fields[3]) + val
                else:
                    combineddictionarybefore1980.update({fields[3]:val})
            else:
                if year >= 1980:
                    if combineddictionaryafter1980.has_key(fields[3]):
                        combineddictionaryafter1980[fields[3]] = combineddictionaryafter1980.get(fields[3]) + val
                    else:
                        combineddictionaryafter1980.update({fields[3]:val})

    combined_percentage = {}
    combine_change = {}
    for k,v in combineddictionarybefore1980.iteritems():
        if (k in combineddictionaryafter1980): 
            percent = float(combineddictionaryafter1980[k] - combineddictionarybefore1980[k])/combineddictionarybefore1980[k]
            change = combineddictionaryafter1980[k] - combineddictionarybefore1980[k]
            combined_percentage.update({k:percent})
            combine_change.update({k:change})

    finalcombinedpercentage = sorted(combined_percentage.items(), key=lambda x: x[1], reverse = True)
    print "Large percentage increase in either Gender"
    print finalcombinedpercentage[0]
    finalcombinedpercentage = sorted(combined_percentage.items(), key=lambda x: x[1])
    print "Large percentage decrease in either Gender"
    print finalcombinedpercentage[0]

    finalcombinedchange = sorted(combine_change.items(), key=lambda x: x[1], reverse = True)
    print "Large  increase in either Gender"
    print finalcombinedchange[0]
    finalcombinedchange = sorted(combine_change.items(), key=lambda x: x[1])
    print "Large  decrease in either Gender"
    print finalcombinedchange[0]

    i = 0


    maledictionaryat1980 = {}
    maledictionaryat2014 = {}    
    with open("outputfile.TXT", "r") as alllines:
        array = []
        for line in alllines:
            fields = line.split(',')
            tupppp = []
            val = int(fields[4])
            year = int(fields[2])
            if year == 1980 and fields[1] == 'M':
                if maledictionaryat1980.has_key(fields[3]):
                    maledictionaryat1980[fields[3]] = maledictionaryat1980.get(fields[3]) + val
                else:
                    maledictionaryat1980.update({fields[3]:val})
            else:
                if year == 2014 and fields[1] == 'M':
                    if maledictionaryat2014.has_key(fields[3]):
                        maledictionaryat2014[fields[3]] = maledictionaryat2014.get(fields[3]) + val
                    else:
                        maledictionaryat2014.update({fields[3]:val})

    combined_percentage = {}
    combine_change = {}
    for k,v in maledictionaryat1980.iteritems():
        if (k in maledictionaryat2014): 
            percent = float(maledictionaryat2014[k] - maledictionaryat1980[k])/maledictionaryat1980[k]
            change = maledictionaryat2014[k] - maledictionaryat1980[k]
            combined_percentage.update({k:percent})
            combine_change.update({k:change})

    finalcombinedpercentage = sorted(combined_percentage.items(), key=lambda x: x[1], reverse = True)
    print "Large percentage increase from 1980 to 2014 in male"
    print finalcombinedpercentage[0]
    finalcombinedpercentage = sorted(combined_percentage.items(), key=lambda x: x[1])
    print "Large percentage decrease from 1980 to 2014 in male"
    print finalcombinedpercentage[0]

    finalcombinedchange = sorted(combine_change.items(), key=lambda x: x[1], reverse = True)
    print "Large increase from 1980 to 2014 in male"
    print finalcombinedchange[0]
    finalcombinedchange = sorted(combine_change.items(), key=lambda x: x[1])
    print "Large decrease from 1980 to 2014 in male"
    print finalcombinedchange[0]



    femaledictionaryat1980 = {}
    femaledictionaryat2014 = {}    
    with open("outputfile.TXT", "r") as alllines:
        array = []
        for line in alllines:
            fields = line.split(',')
            tupppp = []
            val = int(fields[4])
            year = int(fields[2])
            if year == 1980 and fields[1] == 'F':
                if femaledictionaryat1980.has_key(fields[3]):
                    femaledictionaryat1980[fields[3]] = femaledictionaryat1980.get(fields[3]) + val
                else:
                    femaledictionaryat1980.update({fields[3]:val})
            else:
                if year == 2014 and fields[1] == 'F':
                    if femaledictionaryat2014.has_key(fields[3]):
                        femaledictionaryat2014[fields[3]] = femaledictionaryat2014.get(fields[3]) + val
                    else:
                        femaledictionaryat2014.update({fields[3]:val})

    combined_percentage = {}
    combine_change = {}
    for k,v in femaledictionaryat1980.iteritems():
        if (k in femaledictionaryat2014): 
            percent = float(femaledictionaryat2014[k] - femaledictionaryat1980[k])/femaledictionaryat1980[k]
            change = femaledictionaryat2014[k] - femaledictionaryat1980[k]
            combined_percentage.update({k:percent})
            combine_change.update({k:change})

    finalcombinedpercentage = sorted(combined_percentage.items(), key=lambda x: x[1], reverse = True)
    print "Large percentage increase from 1980 to 2014 in female"
    print finalcombinedpercentage[0]
    finalcombinedpercentage = sorted(combined_percentage.items(), key=lambda x: x[1])
    print "Large percentage decrease from 1980 to 2014 in female"
    print finalcombinedpercentage[0]

    finalcombinedchange = sorted(combine_change.items(), key=lambda x: x[1], reverse = True)
    print "Large increase from 1980 to 2014 in female"
    print finalcombinedchange[0]
    finalcombinedchange = sorted(combine_change.items(), key=lambda x: x[1])
    print "Large decrease from 1980 to 2014 in female"
    print finalcombinedchange[0]



    combineddictionaryat1980 = {}
    combineddictionaryat2014 = {}    
    with open("outputfile.TXT", "r") as alllines:
        array = []
        for line in alllines:
            fields = line.split(',')
            tupppp = []
            val = int(fields[4])
            year = int(fields[2])
            if year == 1980:
                if combineddictionaryat1980.has_key(fields[3]):
                    combineddictionaryat1980[fields[3]] = combineddictionaryat1980.get(fields[3]) + val
                else:
                    combineddictionaryat1980.update({fields[3]:val})
            else:
                if year == 2014:
                    if combineddictionaryat2014.has_key(fields[3]):
                        combineddictionaryat2014[fields[3]] = combineddictionaryat2014.get(fields[3]) + val
                    else:
                        combineddictionaryat2014.update({fields[3]:val})

    combined_percentage = {}
    combine_change = {}
    for k,v in combineddictionaryat1980.iteritems():
        if (k in combineddictionaryat2014): 
            percent = float(combineddictionaryat2014[k] - combineddictionaryat1980[k])/combineddictionaryat1980[k]
            change = combineddictionaryat2014[k] - combineddictionaryat1980[k]
            combined_percentage.update({k:percent})
            combine_change.update({k:change})

    finalcombinedpercentage = sorted(combined_percentage.items(), key=lambda x: x[1], reverse = True)
    print "Large percentage increase from 1980 to 2014 in either gender"
    print finalcombinedpercentage[0]
    finalcombinedpercentage = sorted(combined_percentage.items(), key=lambda x: x[1])
    print "Large percentage decrease from 1980 to 2014 in either gender"
    print finalcombinedpercentage[0]

    finalcombinedchange = sorted(combine_change.items(), key=lambda x: x[1], reverse = True)
    print "Large increase from 1980 to 2014 in either gender"
    print finalcombinedchange[0]
    finalcombinedchange = sorted(combine_change.items(), key=lambda x: x[1])
    print "Large decrease from 1980 to 2014 in either gender"
    print finalcombinedchange[0]






    combineddictionaryat1910 = {}
    combineddictionaryat2014 = {}    
    with open("outputfile.TXT", "r") as alllines:
        array = []
        for line in alllines:
            fields = line.split(',')
            tupppp = []
            val = int(fields[4])
            year = int(fields[2])
            if year == 1910:
                if combineddictionaryat1910.has_key(fields[3]):
                    combineddictionaryat1910[fields[3]] = combineddictionaryat1910.get(fields[3]) + val
                else:
                    combineddictionaryat1910.update({fields[3]:val})
            else:
                if year == 2014:
                    if combineddictionaryat2014.has_key(fields[3]):
                        combineddictionaryat2014[fields[3]] = combineddictionaryat2014.get(fields[3]) + val
                    else:
                        combineddictionaryat2014.update({fields[3]:val})

    combined_percentage = {}
    for k,v in combineddictionaryat1910.iteritems():
        if (k in combineddictionaryat2014): 
            percent = float(combineddictionaryat2014[k] - combineddictionaryat1910[k])
            combined_percentage.update({k:percent})

    finalcombinedpercentage = sorted(combined_percentage.items(), key=lambda x: x[1], reverse = True)
    print "Large increase from 1910 to 2014 in either gender"
    print finalcombinedpercentage[0]
    finalcombinedpercentage = sorted(combined_percentage.items(), key=lambda x: x[1])
    print "Large decrease from 1910 to 2014 in either gender"
    print finalcombinedpercentage[0]

    End()


    