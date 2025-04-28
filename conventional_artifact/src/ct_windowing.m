
function [in_img_win] = ct_windowing(in_img, windowing)
    in_img_win = in_img;
    if strcmp (windowing, 'airtools_soft_artifact')
      % it is different than soft windowing
      LL = 1075-700/2;
      UL = 1075+700/2;
    elseif strcmp (windowing, 'soft')
      LL = 1075-400/2;
      UL = 1075+400/2;
    elseif strcmp (windowing, 'irt_soft_artifact')
      % it is different than soft windowing
      LL = 1286-780/2;
      UL = 1286+780/2;
    elseif strcmp (windowing, 'L_105_W_800')
      % it is different than soft windowing
      LL = 1130-800/2;
      UL = 1130+800/2;
    elseif strcmp(windowing, 'bone')
      LL = 425-1500/2;
      UL = 425 + 1500/2;
    end 
    LL_ind             = find(in_img<=LL);
    in_img_win(LL_ind) = LL;
    UL_ind             = find(in_img>=UL);
    in_img_win(UL_ind) = UL;
end
