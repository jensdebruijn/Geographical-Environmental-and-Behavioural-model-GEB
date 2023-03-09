# -*- coding: utf-8 -*-
import os
import json
import requests
import geopandas as gpd
from arcgis2geojson import arcgis2geojson

from preconfig import ORIGINAL_DATA, INPUT

COMMAND_AREAS_DIR = os.path.join(ORIGINAL_DATA, 'command_areas')
if not os.path.exists(COMMAND_AREAS_DIR):
    os.makedirs(COMMAND_AREAS_DIR)


def download(IDs: list[int]) -> None:
    """Downloads command areas of given IDs from India-WRIS API and saves to features folder
    
    Args:
        IDs: list of command area IDs"""
    download_folder = os.path.join(COMMAND_AREAS_DIR, 'features')
    os.makedirs(download_folder, exist_ok=True)

    n = len(IDs)
    for i, ID in enumerate(IDs, start=1):
        print(f"{i}/{n}", end='\r')
        fn = os.path.join(COMMAND_AREAS_DIR, 'features', f'{ID}.json')
        if not os.path.exists(fn):
            url = f"https://gis.indiawris.gov.in/server/rest/services/SubInfoSysLCC/WRP_Old/MapServer/7/query?where=&text=&objectIds={ID}&time=&geometry=&geometryType=esriGeometryEnvelope&inSR=&spatialRel=esriSpatialRelIntersects&relationParam=&outFields=type%2Cstatus%2Cstate%2Ccca%2Ctsp%2Cdpa%2Clulc_st%2Cpop_st%2Cdist_benf&returnGeometry=true&returnTrueCurves=false&maxAllowableOffset=&geometryPrecision=&outSR=&returnIdsOnly=false&returnCountOnly=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&returnZ=false&returnM=false&gdbVersion=&returnDistinctValues=false&resultOffset=&resultRecordCount=&queryByDistance=&returnExtentsOnly=false&datumTransformation=&parameterValues=&rangeValues=&f=json"
            geojson = requests.get(url)
            geojson = geojson.content.decode()
            obj = json.loads(geojson)
            if 'error' in obj:
                print("error for", fn)
            else:
                with open(fn, 'w') as f:
                    f.write(geojson)


def merge_and_export() -> None:
    """Converts all arcgis files to geojson, and merges in GeoJSON feature collection"""
    feature_collection = {
        'type': 'FeatureCollection',
        'features': []
    }
    files = os.listdir(os.path.join(COMMAND_AREAS_DIR, 'features'))
    n_files = len(files)
    for i, fn in enumerate(files, start=1):
        print(f"converting file {i}/{n_files}", end='\r')
        fn = os.path.join(COMMAND_AREAS_DIR, 'features', fn)
        with open(fn, 'r') as f:
            geojson = arcgis2geojson(json.load(f))
            feature_collection['features'].extend(geojson['features'])
        break
    
    gdf = gpd.GeoDataFrame.from_features(feature_collection)
    gdf = gdf.set_crs("""PROJCS["WGS_1984_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["False_Easting",4000000.0],PARAMETER["False_Northing",4000000.0],PARAMETER["Central_Meridian",80.0],PARAMETER["Standard_Parallel_1",12.4729444],PARAMETER["Standard_Parallel_2",35.17280555],PARAMETER["Latitude_Of_Origin",24.0],UNIT["Meter",1.0]]""", allow_override=True)
    gdf = gdf.to_crs("EPSG:4326")
    folder = os.path.join(ORIGINAL_DATA, 'command_areas')
    os.makedirs(folder, exist_ok=True)
    gdf.to_file(os.path.join(folder, 'command_areas.shp'))

if __name__ == '__main__':
    IDs = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,77,88,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,177,178,179,180,181,182,183,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,281,282,283,284,285,286,287,288,289,290,291,293,294,295,296,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,315,316,317,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,379,380,381,382,2560,2748,2762,2763,2764,2771,2772,2773,2774,2779,2780,2782,2785,2786,2787,2788,2790,2792,2794,2798,2799,2800,2801,2802,2803,2804,2806,2807,2810,2811,2812,2813,2814,2815,2817,2819,2820,2821,2822,2823,2824,2826,2830,2831,2832,2834,2835,2843,2844,2846,2848,2849,2850,2851,2852,2853,2854,2855,2856,2857,2858,2859,2860,2861,2862,2863,2864,2865,2866,2867,2868,2869,2870,2871,2872,2874,2875,2876,2877,2878,2879,2880,2881,2882,2883,2884,2886,2889,2890,2891,2892,2893,2894,2895,2896,2897,2899,2900,2901,2902,2903,2904,2905,2906,2907,2910,2912,2913,2914,2915,2916,2917,2920,2921,2922,2923,2924,2925,2926,2929,2931,2932,2933,2934,2935,2941,2944,2948,2949,2950,2952,2964,2965,2968,2987,2988,2989,2990,2991,3144,3145,3146,3147,3148,3149,3150,3151,3154,3160,3161,3162,3163,3164,3165,3166,3167,3168,3169,3170,3171,3172,3173,3174,3175,3176,3177,3178,3179,3180,3181,3182,3183,3184,3185,3186,3187,3188,3189,3190,3191,3192,3193,3194,3195,3196,3197,3198,3199,3200,3201,3202,3203,3204,3205,3206,3207,3208,3209,3210,3211,3212,3214,3215,3216,3217,3218,3219,3220,3221,3222,3223,3225,3226,3227,3228,3229,3230,3231,3232,3233,3234,3235,3236,3237,3238,3239,3240,3241,3242,3243,3244,3245,3246,3248,3249,3250,3251,3252,3253,3254,3255,3256,3258,3259,3260,3261,3262,3301,3302,3303,3304,3305,3306,3307,3308,3333,3410,3411,3426,3427,3428,3429,3430,3432,3433,3434,3435,3436,3437,3438,3439,3440,3441,3442,3445,3446,3447,3448,3449,3451,3455,3456,3457,3459,3460,3462,3463,3464,3465,3466,3468,3469,3470,3471,3472,3473,3475,3477,3481,3482,3484,3485,3486,3487,3488,3489,3490,3491,3494,3495,3496,3497,3498,3499,3500,3501,3503,3504,3505,3506,3508,3509,3512,3514,3516,3519,3520,3522,3523,3524,3525,3526,3528,3530,3531,3532,3533,3534,3535,3536,3537,3540,3541,3542,3544,3545,3546,3547,3548,3549,3550,3551,3552,3554,3555,3557,3559,3560,3561,3562,3563,3565,3566,3567,3570,3571,3573,3575,3576,3578,3579,3581,3583,3584,3585,3587,3588,3590,3591,3592,3593,3597,3598,3599,3600,3601,3603,3604,3605,3606,3607,3608,3609,3610,3614,3615,3616,3617,3618,3619,3620,3621,3622,3623,3624,3625,3626,3627,3628,3629,3630,3631,3632,3633,3634,3635,3636,3637,3638,3640,3641,3642,3643,3644,3646,3647,3648,3649,3650,3651,3652,3653,3654,3655,3656,3657,3658,3659,3660,3661,3662,3663,3664,3665,3666,3667,3668,3669,3670,3671,3672,3673,3674,3675,3677,3678,3679,3680,3682,3683,3684,3685,3686,3687,3688,3690,4089,4090,4091,4095,4096,4097,4098,4099,4100,4101,4102,4131,4132,4133,4134,4135,4136,4415,4416,4418,4421,4509,4510,4511,4512,4513,4514,4515,4516,4517,4518,4519,4520,4521,4522,4524,4525,4526,4527,4528,4529,4530,4531,4532,4533,4534,4535,4536,4537,4538,4539,4540,4541,4542,4543,4544,4545,4546,4547,4548,4549,4550,4551,4552,4553,4554,4555,4556,4557,4558,4559,4560,4561,4562,4563,4564,4565,4566,4567,4568,4569,4570,4571,4572,4573,4574,4575,4576,4577,4578,4579,4580,4581,4583,4584,4585,4586,4587,4588,4589,4590,4591,4592,4593,4594,4595,4596,4597,4598,4599,4600,4601,4602,4603,4604,4605,4606,4607,4608,4609,4610,4611,4612,4613,4614,4615,4616,4617,4618,4619,4620,4621,4622,4623,4624,4625,4627,4628,4630,4631,4632,4633,4634,4635,4637,4638,4639,4640,4641,4642,4643,4644,4645,4646,4647,4648,4649,4650,4651,4652,4653,4654,4655,4656,4657,4658,4659,4660,4661,4662,4663,4664,4665,4666,4667,4668,4669,4670,4671,4672,4673,4674,4675,4676,4677,4678,4679,4680,4681,4682,4683,4684,4685,4686,4687,4688,4689,4690,4691,4692,4693,4694,4695,4696,4697,4698,4699,4700,4701,4702,4703,4704,4705,4706,4707,4708,4709,4710,4711,4712,4713,4714,4715,4716,4717,4718,4719,4720,4721,4722,4723,4724,4725,4726,4727,4728,4729,4730,4731,4732,4733,4734,4735,4736,4737,4738,4739,4740,4741,4742,4743,4744,4745,4747,4748,4749,4750,4751,4752,4753,4754,4755,4756,4757,4758,4759,4760,4761,4762,4763,4764,4765,4766,4767,4769,4770,4771,4772,4773,4774,4775,4776,4777,4778,4779,4780,4781,4782,4783,4784,4785,4786,4787,4788,4789,4790,4791,4792,4793,4794,4795,4796,4797,4798,4799,4800,4801,4802,4803,4804,4805,4806,4807,4808,4809,4810,4811,4812,4813,4814,4815,4816,4817,4818,4819,4820,4821,4822,4823,4824,4825,4826,4827,4828,4829,4830,4831,4832,4833,4834,4835,4836,4837,4838,4839,4840,4841,4842,4843,4844,4845,4846,4847,4848,4849,4850,4851,4852,4853,4854,4855,4856,4857,4858,4859,4860,4861,4862,4863,4864,4865,4866,4867,4868,4869,4870,4871,4872,4873,4874,4875,4876,4877,4878,4879,4880,4881,4882,4883,4884,4885,4886,4887,4888,4889,4890,4891,4892,4893,4894,4895,4896,4897,4898,4899,4900,4901,4902,4940,4941,4942,4943,4944,4945,4946,4947,4948,4949,4950,4951,4952,4953,4954,4955,4956,4957,4958,4959,4960,4961,4962,4963,4964,4965,4966,4967,4968,4969,4970,4971,4972,4973,4974,4975,4976,4977,4978,4979,4980,4981,4982,4983,4984,4985,4987,4988,4989,4990,4991,4992,4993,4994,4995,4996,4997,4998,4999,5000,5001,5002,5003,5004,5005,5006,5007,5009,5010,5011,5012,5013,5014,5015,5016,5017,5018,5019,5021,5022,5023,5024,5025,5026,5027,5028,5029,5030,5031,5032,5033,5034,5035,5036,5037,5038,5039,5040,5041,5042,5043,5044,5045,5046,5047,5048,5049,5050,5051,5052,5053,5054,5055,5056,5057,5058,5059,5060,5061,5062,5063,5064,5065,5066,5067,5068,5069,5070,5072,5073,5074,5075,5076,5077,5078,5079,5080,5081,5082,5083,5084,5085,5086,5087,5088,5089,5090,5091,5092,5093,5094,5095,5096,5097,5098,5099,5100,5101,5102,5103,5104,5105,5106,5107,5108,5109,5110,5111,5112,5113,5114,5115,5116,5117,5118,5119,5120,5121,5122,5123,5124,5125,5126,5127,5128,5129,5130,5131,5132,5133,5134,5135,5136,5137,5138,5139,5141,5142,5143,5144,5145,5146,5147,5148,5149,5150,5151,5152,5153,5154,5155,5156,5157,5158,5159,5160,5161,5162,5163,5164,5165,5166,5167,5168,5169,5170,5171,5172,5173,5174,5175,5176,5177,5178,5179,5180,5181,5182,5183,5184,5185,5186,5187,5188,5189,5190,5191,5192,5193,5194,5195,5196,5197,5198,5199,5200,5201,5202,5203,5204,5205,5206,5207,5208,5209,5210,5211,5212,5213,5214,5215,5216,5217,5327,5538,5539,5540,5541,5542,5543,5544,5545,5546,5547,5548,5549,5550,5551,5552,5553,5554,5555,5556,5557,5558,5559,5560,5561,5562,5563,5564,5565,5566,5567,5568,5569,5570,5571,5572,5573,5574,5575,5576,5577,5578,5579,5580,5581,5582,5583,5584,5585,5586,5587,5588,5589,5590,5591,5592,5593,5594,5595,5596,5597,5598,5599,5600,5601,5877,5878,5879,5880,5881,5882,5883,5884,5885,5886,5887,5888,5889,5890,5891,5892,5893,5894,5895,5896,5897,5898,5899,5900,5901,5902,5903,5904,5905,5906,5907,5908,5909,5910,5911,5912,5913,5914,5915,5916,5917,5918,5919,5920,5921,5922,5923,5924,5925,5926,5927,5928,5929,5930,5931,5932,5933,5934,5935,5936,5937,5938,5939,5940,5941,5942,5943,5944,5945,5946,5947,5948,5949,5950,5951,5952,5953,5954,5955,5956,5957,5958,5959,5960,5961,5962,5964,5965,5966,5967,5968,5969,5970,5971,5972,5974,5975,6469,6473,6474,6475,6476,6478,6480,6482,6485,6486,6487,6488,6489,6490,6491,6492,6493,6495,6497,6498,6499,6500,6501,6502,6503,6504,6505,6506,6507,6508,6509,6510,6511,6512,6513,6514,6515,6516,6517,6518,6519,6521,6522,6523,6524,6525,6526,6527,6528,6529,6530,6531,6532,6533,6534,6535,6536,6537,6538,6539,6540,6541,6544,6545,6546,6547,6548,6550,6551,6552,6553,6554,6555,6556,6558,6559,6563,6565,6566,6567,6569,6571,6572,6578,6579,7066,7067,7068,7069,7070,7071,7391,7392,7711,7727,7728,7729,7730,7731,7732,7733,7734,7735,7736,7737,7738,8031,8032,8351,8354,8356,8357,8361,8365,8687,8991,8992,8993,8994,8995,8996,8997,8998,8999,9000,9001,9002,9003,9004,9005,9006,9007,9008,9009,9010,9011,9012,9013,9014,9015,9016,9017,9018,9019,9020,9021,9022,9023,9024,9025,9026,9027,9028,9029,9030,9031,9032,9033,9034,9035,9037,9038,9039,9040,9041,9042,9043,9044,9045,9046,9047,9048,9049,9050,9051,9052,9053,9054,9055,9056,9057,9058,9059,9060,9061,9062,9063,9064,9065,9066,9067,9068,9311,9315,9631,9632,9633,9634,9635,9636,9637,9638,9639,9640,9641,9642,9643,9644,9645,9646,9647,9648,9649,9650,9651,9652,9653,9654,9655,9656,9657,9658,9659,9660,9661,9662,9663,9664,9665,9666,9667,9668,9669,9670,9671,9672,9673,9674,9675,9676,9677,9678,9679,9680,9681,9682,9683,9684,9685,9686,9687,9688,9689,9690,9691,9692,9693,9694,9695,9696,9697,9698,9699,9700,9701,9702,9703,9704,9705,9706,9707,9708,9709,9710,9711,9712,9713,9714,9715,9716,9717,9718,9719,9720,9721,9722,9723,9724,9725,9726,9727,9728,9729,9730,9731,9732,9733,9734,9735,9736,9737,9738,9739,9740,9741,9742,9951,10271,10272,10273,10275,10286,10287,10289,10290,10291,10292,10591,10592,10593,10594,10595,10596,10597,10598,10599,10600,10601,10602,10603,10604,10605,10606,10607,10608,10609,10610,10611,10612,10613,10614,10615,10616,10617,10618,10619,10620,10621,10622,10623,10624,10625,10626,10627,10628,10629,10630,10631,10632,10633,10634,10635,10636,10637,10638,10639,10640,10641,10642,10643,10644,10645,10646,10647,10648,10649,10650,10651,10652,10653,10654,10655,10656,10657,10658,10659,10660,10661,10662,10663,10664,10665,10666,10667,10668,10669,10670,10671,10672,10673,10674,10675,10676,10677,10678,10680,10681,10682,10683,10684,10685,10686,10687,10688,10689,10690,10691,10692,10693,10694,10695,10696,10697,10698,10699,10700,10701,10702,10703,10704,10705,10706,10707,10708,10709,10710,10711,10712,10713,10714,10715,10716,10717,10718,10719,10720,10721,10722,10723,10724,10725,10726,10727,10728,10729,10730,10732,10733,10734,10735,10736,10737,10738,10739,10740,10741,10742,10743,10744,10745,10746,10747,10748,10749,10750,10751,10752,10753,10754,10755,10756,10757,10758,10759,10760,10761,10762,10763,10764,10765,10766,10767,10768,10769,10770,10771,10772,10773,10774,10775,10776,10777,10778,10779,10780,10781,10782,10783,10784,10785,10786,10787,10788,10789,10790,10792,10793,10794,10795,10796,10797,10798,10799,10800,10801,10802,10803,10804,10805,10806,10807,10808,10809,10810,10811,10812,10813,10814,10815,10816,10817,10818,10819,10820,10821,10822,10823,10824,10825,10826,10827,10828,10829,10830,10831,10832,10833,10834,10835,10836,10837,10838,10839,10840,10841,10842,10843,10844,10845,10846,10847,10848,10849,10850,10851,10852,10853,10854,10855,10856,10857,10858,10859,10860,10861,10862,10863,10864,10865,10871,10872,10873,10874,10875,10876,10877,10878,10879,10880,10881,10882,10883,10884,10886,10887,10888,10889,10890,10891,10892,10893,10894,10895,10896,10897,10898,10899,10900,10901,10902,10903,10904,10905,10906,10907,10908,10909,10910,10911,10912,10913,10914,10915,10916,10917,10918,10919,10920,10921,10922,10923,10924,10925,10926,10927,10928,10929,10930,10931,10932,10933,10934,10935,10936,10937,10938,10939,10940,10941,10942,10943,10944,10945,10946,10947,10948,10949,10950,10951,10952,10953,10954,10955,10956,10957,10958,10959,10960,10961,10962,10963,10964,10965,10966,10967,10968,10969,10970,10971,10972,10973,10974,10975,10976,10978,10979,10980,10981,10982,10983,10984,10985,10986,10987,10988,10989,10990,10991,10992,10993,10994,10995,10996,10997,10998,10999,11000,11001,11002,11003,11004,11005,11006,11007,11008,11009,11010,11011,11012,11013,11014,11015,11016,11017,11018,11019,11020,11021,11022,11023,11024,11025,11026,11028,11029,11030,11031,11032,11033,11034,11035,11036,11037,11038,11039,11040,11041,11042,11043,11044,11045,11046,11047,11048,11049,11050,11051,11052,11053,11054,11055,11056,11057,11058,11059,11060,11061,11062,11063,11064,11066,11067,11068,11069,11070,11071,11072,11073,11074,11075,11077,11078,11079,11080,11082,11083,11085,11086,11087,11088,11089,11090,11091,11092,11093,11094,11095,11096,11097,11098,11099,11100,11101,11102,11103,11104,11105,11106,11107,11108,11109,11110,11111,11112,11113,11114,11115,11116,11117,11118,11119,11120,11121,11122,11123,11124,11125,11126,11127,11128,11129,11130,11235,11236,11237,11238,11239,11240,11241,11242,11243,11244,11245,11246,11247,11248,11249,11250,11551,11552,11553,11554,11555,11556,11557,11558,11559,11560,11561,11562,11563,11564,11565,11566,11567,11568,11569,11570,11571,11572,11573,11574,11575,11577,11578,11579,11580,11581,11582,11583,11585,11586,11587,11588,11589,11590,11591,11592,11593,11594,11595,11596,11597,11598,11599,11600,11871,11872,11873,11874,11875,11876,11877,11878,11880,11881,11882,11887,11889,11892,11904,11908,11909,12202,12525,12526,12527,12529,12530,12531,12532,12534,12535,12537,12538,12542,12543,12545,12842,13802,13803,13806,13807,13818,13819,13820,14122,14123,14124,14125,14127,14128,14130,14136,14442,14454,14460,14465,14762,14763,14838,14844,14845,14847,14848,14853,14854,14855,14856,14862,14870,14882,14886,14888,14891,14892,15082,15083,15084,15085,15086,15087,15088,15089,15092,15102,15103,15105,15106,15107,15108,15118,15119,15120,15121,15402,15403,15404,15405,15406,15407,15408,15410,15411,15412,15413,15414,15415,15416,15417,15419,15722,15724,15728,15729,15730,15732,15733,15734,16042,16043,16048,16052,16053,16054,16057,16058,16059,16060,16061,16062,16064,16065,16066,16067,16068,16070,16362,16363,16364,16366,16367,16368,16371,16683,17322,17323,17324,17325,17326,17327,17330,17331,17332,17333,17334,17335,17336,17337,17338,17339,17340,17341,17642,17644,17646,17647,17648,17649,17654,17655,17657,17962,17963,17964,17965,17966,17967,17968,17969,17970,17971,17972,17973,17974,17975,17976,17977,17978,17979,17980,17981,17982,17983,17984,17985,17986,17987,17988,17989,17990,17991,17992,17993,17994,17995,17996,17997,17998,17999,18000,18001,18002,18003,18004,18005,18006,18007,18008,18009,18010,18011,18013,18014,18018,18019]
    # download(IDs)
    merge_and_export()