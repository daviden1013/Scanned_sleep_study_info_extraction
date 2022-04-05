require(magrittr)
require(readxl)
require(data.table)
require(stringi)
require(glue)

getScriptPath <- function() {
    out <- try(sys.frame(1)$ofile, silent = T)
    if (is.null(out) | inherits(out, "try-error")) {
        dirname(rstudioapi::getActiveDocumentContext()$path)
    } else {
        dirname(out)
    }
}

scriptPath <- getScriptPath()
setwd(scriptPath)
#sourcePath <- "./textgray"

sourcePath <- "../ersn01cntrst60/"
targetPath <- "./ersn01cntrst60"
if (!dir.exists(targetPath)) dir.create(targetPath)


# Create data.frame csv filenames stored in fileID variable
txtNames <- dir(sourcePath, pattern = ".*csv", https://urldefense.proofpoint.com/v2/url?u=http-3A__ignore.case&d=DwIGAg&c=bKRySV-ouEg_AT-w2QWsTdd9X__KYh9Eq2fdmQDVZgw&r=uQnfl4BWxOwiq5m1EPdUhsRAZBhKvcvjyaxeWSIcKsI&m=JGTMUr5cWp9_C7QiCQUY3PXSemTudFzOTOUjFgHPZOM&s=-iPrI9e9svBKNsfuUhe0A7WfpaCdpPuF-n9m8QKXp_w&e=  = T)
dat <- data.frame(fileID = txtNames)

# Capture MRN from filenames and store it in MRN variable
MRNreg <- "\\d{6}[[:alpha:]]{1}"
dat$MRN <-sub(glue("^.+({MRNreg}).+$"), "\\1", dat$fileID)
keep <- grep(MRNreg, dat$MRN)
dat[-keep,]
dat <- dat[keep, ]


# Use md5 to encrypt MRN and create new variables with filenames and label with
# MRN substituted with encrypted MRN
dat$encryptMRN <- openssl::md5(dat$MRN)
dat$encryptFname <- dat$fileID
dat$lab = dat$encryptLab <- sub("^(\\d{4})-(.{7,})-.+$","\\1-\\2", dat$fileID)
for (i in seq(nrow(dat))) {
    dat$encryptLab[[i]] <- with(dat, sub(MRN[[i]], encryptMRN[[i]], encryptLab[[i]]))
    dat$encryptFname[[i]] <- with(dat, sub(MRN[[i]], encryptMRN[[i]], encryptFname[[i]]))
}

# Read file with patients' names and MRN and merge to main dat
nams <- readxl::read_xlsx("./export.xlsx", sheet = 1) %>%
    setNames(c("MRN", "name")) %>%
    as.data.frame()

dat <- merge(dat, nams, by = "MRN", all = T)
dat <- dat[order(dat$fileID), ]
lapply(dat, function(x) sum(is.na(x)))

id <- is.na(dat$name)
dat[id, ]
dat <- dat[!id, ]

# Create additional variables which captures first and last name of each patient
dat$patn <- sub("^(\\w+,\\w+)\\s(.*)$", "\\1", dat$name)
dat[c("lastn", "firstn")] <-
    dat$patn %>%
    strsplit(",") %>%
    do.call(rbind, .)

# Regular expression which captures date formats in text
dateRegEx <- "^\\s*\\d{1,2}\\s*[/.-]\\s*\\d{1,2}\\s*[/.-]\\s*\\d{4}"

# define function to remove dates, MRN, and patient names from csv files
removePersonalInfo <- function(x, info) {
    if (grepl(dateRegEx, x)) {
        x <- "[DATE]"
    } else if (grepl(info[["MRN"]], x)) {
        x <- "[PATID]"
    } else {
        x <- sub(info[["lastn"]], "[PATNAME]", x, https://urldefense.proofpoint.com/v2/url?u=http-3A__ignore.case&d=DwIGAg&c=bKRySV-ouEg_AT-w2QWsTdd9X__KYh9Eq2fdmQDVZgw&r=uQnfl4BWxOwiq5m1EPdUhsRAZBhKvcvjyaxeWSIcKsI&m=JGTMUr5cWp9_C7QiCQUY3PXSemTudFzOTOUjFgHPZOM&s=-iPrI9e9svBKNsfuUhe0A7WfpaCdpPuF-n9m8QKXp_w&e=  = T)
        x <- sub(info[["firstn"]], "[PATNAME]", x, https://urldefense.proofpoint.com/v2/url?u=http-3A__ignore.case&d=DwIGAg&c=bKRySV-ouEg_AT-w2QWsTdd9X__KYh9Eq2fdmQDVZgw&r=uQnfl4BWxOwiq5m1EPdUhsRAZBhKvcvjyaxeWSIcKsI&m=JGTMUr5cWp9_C7QiCQUY3PXSemTudFzOTOUjFgHPZOM&s=-iPrI9e9svBKNsfuUhe0A7WfpaCdpPuF-n9m8QKXp_w&e=  = T)
    }
    return(x)
}


# create in target path de-identified csv files with filename where MRN is encrypted
infoTab <- dat[c("encryptFname", "fileID", "MRN", "lastn", "firstn")] %>% as.matrix()
for (entry in seq(nrow(infoTab))) {
    sourceFile <- paste(sourcePath, infoTab[entry, "fileID"], sep = "/")
    targetFile <- sub("_", "-", paste(targetPath, infoTab[entry, "encryptFname"], sep = "/"))
    txtDat <- data.table::fread(file = sourceFile)
    info <- infoTab[entry, ]

    for (i in seq(nrow(txtDat))) {
        txtDat$text[[i]] <- removePersonalInfo(txtDat$text[[i]], info)
    }
    data.table::fwrite(txtDat, targetFile)
}


labels <- readxl::read_xls("./de_identify/label (filled).xls", sheet = 1) %>% as.data.frame()
dictLab <- dat[c("lab", "encryptLab")][!duplicated(dat[c("lab", "encryptLab")]), ]
write.table(labs, "./de_identify/encryptedLabel.csv", row.names = F, sep = ",")

dict <- dat[!duplicated(dat$MRN), c("encryptMRN","MRN")]
dict <- with(dict, setNames(MRN, encryptMRN))
saveRDS(dict, "./de_identify/dict.RDS")







