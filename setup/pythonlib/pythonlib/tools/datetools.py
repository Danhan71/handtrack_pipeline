    # # how to compare dates
    # import datetime
    # d = datetime.datetime.strptime(str(date), "%y%m%d")

    # d1 = datetime.datetime.today()
    # d<d1


def datenum2obj(date, dtformat="%y%m%d"):
    """ converts date (in str or int) to a datetime object
    - by default, date in format yymmdd (e..g, 200130) 
    - can be string or int
    - returns datetime obnject"""
    from datetime import datetime
    
    if isinstance(date, int):
        dt = datetime.strptime(str(date), dtformat)
    elif isinstance(date, str):
        dt = datetime.strptime(date, dtformat)
    else:
        print(date)
        assert False, "what type is this (date)?"
    return dt
    

def getDateList(sdate=None, edate=None):
    """ get list of dates between (including) sdate and edate
    - dates can either be None (defualt, see below), strings, or
    ints (in format yymmdd).
    - default, sdate is 1/1/20;
    - default, edate is today.
    - date format will be yymmdd (e..g, 200130)
    """
#     sdate = date(2020, 1, 1)   # start dat
    from datetime import date, timedelta, datetime
    if sdate is None:
        sdate=datetime(2020, 1, 1)
    else:
        sdate = datenum2obj(sdate)
        
    if edate is None:
        edate =  datetime.today()
    else:
        edate = datenum2obj(edate)
        
    delta = edate - sdate       # as timedelta
    date_list = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    # for i in range(delta.days + 1):
    #     day = sdate + timedelta(days=i)
    #     print(day)
    date_list = [d.strftime("%y%m%d") for d in date_list]
    return date_list

def standardizeTime(datetime, datestart, daystart=9/24, dayend=22/24, staybounded=True):
    """ get a universal "time" value, which is within-experiemnt.
    based on day of ext. e.g., 1.0 means start of first day and 1.99 
    is end of first day. scale will be identical across days. to do this,
    will first figure out the longest day, then use that tos cale. [NOT DONE]
    - by default just uses window definde by daystart and dayend, 
    since that is longer than any day.
    - dates are in format YYMMDD-HHMMSS, strings
    - daystart and dayend are what times will be stretched to, these are 
    in units of day (frac of day). 
    """
    # from pythonlib.tools.datetools import datenum2obj
    
    dt = datenum2obj(datetime, "%y%m%d-%H%M%S")
    dt_start = datenum2obj(datestart, "%y%m%d-%H%M%S")
    
    dt_diff = dt-dt_start
    
    d = dt_diff.days
    s = dt_diff.seconds
    
    # convert s to frac of day
    dayfrac = s/(24*60*60)
    
    # renormalize dayfrac
    dayfrac = (dayfrac - daystart)/(dayend - daystart)
    
    # check that fraqcdya is not outside day bounds.
    if dayfrac<0 or dayfrac>1:
        if staybounded:
            print(datetime, datestart, dayfrac)
            assert False, "this time is otuside bounds.."
        
    return d + dayfrac
