.SECONDARY:

include CONFIG



DOLMAX = 30
R = 0.5

SMOOTH = 0.01


ANGLE_PRECISION = 1
MIN_DISTANCE = 200
THRESH_IOU = 0.1
TIMESPLIT = 3600

TRACK_TOLERANCE_DEGREES_LONGEST = 0.5
THRESH_SLOPE = 0.001
HIT_TOLERANCE = 0
THRESH_BORDER= 0.1
MODEL=mean

FOLDER_TRAJS = $(FOLDER)/trajs
FOLDER_TRAJS_MAN = $(FOLDER)/trajs_man_$(TIMESPLIT)
#/disk2/newjson/detected_alpha_mean_slope_3600_0.01_0.5_0.1_0.001_0.1_0
FOLDER_DETECTED = $(FOLDER)/detected_alpha_$(MODEL)_slope_$(TIMESPLIT)_$(SMOOTH)_$(TRACK_TOLERANCE_DEGREES_LONGEST)_$(THRESH_IOU)_$(THRESH_SLOPE)_$(THRESH_BORDER)_$(HIT_TOLERANCE)
FOLDER_DETECTEDREF = $(FOLDER)/detectedref_alpha_$(MODEL)_slope_$(TIMESPLIT)_$(SMOOTH)_$(ANGLE_PRECISION)_$(MIN_DISTANCE)_$(THRESH_IOU)_$(THRESH_BORDER)


FILES_TRAJS = $(shell cd $(FOLDER_TRAJS);  find -type f,l | grep .parquet$ | cut -c 3-)

FILES_DETECTED = $(foreach f, $(FILES_TRAJS), $(FOLDER_DETECTED)/$(f))
FILES_DETECTEDREF = $(foreach f, $(FILES_TRAJS), $(FOLDER_DETECTEDREF)/$(f))


#first: data $(FOLDER)/detectedref.parquet

first: $(FOLDER)/detectedref.parquet

second: $(FILES_DETECTED)

third: $(FILES_DETECTEDREF)


$(FOLDER)/detectedref.parquet:
	python3 build_json_parquet.py -jsonfolderin $(FOLDER_DETECTEDREF_JSON) -parquetout $@

data:
	mkdir -p $(FOLDER_TRAJS)
	python3 download_data.py -foldertrajs $(FOLDER_TRAJS)

$(FOLDER_DETECTED)/%.parquet: $(FOLDER_TRAJS_MAN)/%.parquet
	mkdir -p $(FOLDER_DETECTED)
	python3 process_airac.py -trajsin $^ -detectedout $@ longest -timesplit $(TIMESPLIT) -smooth $(SMOOTH) -thresh_border $(THRESH_BORDER) -thresh_iou $(THRESH_IOU) -track_tolerance_degrees $(TRACK_TOLERANCE_DEGREES_LONGEST) -thresh_slope $(THRESH_SLOPE) -hit_tolerance $(HIT_TOLERANCE) -model $(MODEL)

$(FOLDER_DETECTEDREF)/%.parquet: $(FOLDER_TRAJS_MAN)/%.parquet
	mkdir -p $(FOLDER_DETECTEDREF)
	python3 process_airac.py -trajsin $^ -detectedout $@ aligned  -timesplit $(TIMESPLIT) -smooth $(SMOOTH) -thresh_border $(THRESH_BORDER) -angle_precision $(ANGLE_PRECISION) -min_distance $(MIN_DISTANCE) -thresh_iou $(THRESH_IOU) -flightplans $(FOLDER)/detectedref.parquet  -model $(MODEL)

$(FOLDER_TRAJS_MAN)/%.parquet: $(FOLDER_TRAJS)/%.parquet
	mkdir -p $(FOLDER_TRAJS_MAN)
	python3 filter_trajs.py -flightplans $(FOLDER)/detectedref.parquet -trajsin $^ -trajsout $@ -timesplit $(TIMESPLIT)

figurespdf:
	python3 detect_longest.py -timesplit $(TIMESPLIT) -smooth $(SMOOTH) -thresh_iou $(THRESH_IOU) -track_tolerance_degrees $(TRACK_TOLERANCE_DEGREES_LONGEST) -thresh_slope $(THRESH_SLOPE) -thresh_border $(THRESH_BORDER) -folderfigures $(FOLDER_FIGURES)  -model $(MODEL) -r $(R) -dolmax $(DOLMAX)
	python3 plot_catalogue.py -folderfigures $(FOLDER_FIGURES)
	python3 compute_stats.py -folderfigures $(FOLDER_FIGURES) -conflict $(FOLDER)/detectedref.parquet -detectedref $(FOLDER_DETECTEDREF) -detectedother $(FOLDER_DETECTED) -r $(R) -dolmax $(DOLMAX)
#	python3 detect_classic.py -timesplit $(TIMESPLIT) -smooth $(SMOOTH) -angle_precision $(ANGLE_PRECISION) -min_distance $(MIN_DISTANCE) -thresh_iou $(THRESH_IOU) -flightplans $(FOLDER)/detectedref.parquet -thresh_border $(THRESH_BORDER) -folderfigures $(FOLDER_FIGURES) -model $(MODEL)


